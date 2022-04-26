import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import Callable, Iterable, Optional

from _base_components import DecoderVEGA


class VEGAVAE(BaseModuleClass):
    """
    VEGA: VAE Enhanced by Gene Annotations model.

    Parameters
    ----------

    """
    def __init__(
        self,
        mask: np.ndarray,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 800,
        n_layers: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        z_dropout: float = 0.,
        positive_decoder: bool = False,
        regularizer: Literal["mask", "l1", "gelnet"] = "mask",
        reg_kwargs: dict = None,
        log_variational: bool = False,
        latent_distribution: str = 'normal',
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder","decoder","none","both"] = "encoder",
        use_layer_norm: Literal["encoder","decoder","none","both"] = "none",
        kl_weight: float = 5e-5,
    ):
        super().__init__()
        
        #self.n_input = mask.shape[0]
        #self.n_gmvs = mask.shape[1]
        n_input = mask.shape[0]
        n_gmvs = mask.shape[1]

        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent = n_gmvs
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.kl_weight = kl_weight
        self.z_dropout = z_dropout
        self.positive_decoder = positive_decoder
        self.regularizer = regularizer
        self.reg_kwargs = reg_kwargs
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        
        # Encoder for GMV activity
        self.z_encoder = Encoder(
            n_input_encoder,
            n_gmvs,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=torch.nn.ReLU,
            )

        # Add dropout to mean and var of z_encoder
        self.z_encoder.mean_encoder = torch.nn.Sequential(self.z_encoder.mean_encoder, torch.nn.Dropout(p=z_dropout))
        self.z_encoder.var_encoder = torch.nn.Sequential(self.z_encoder.var_encoder, torch.nn.Dropout(p=z_dropout))

        # Decoder Gaussian -- add continuous covs
        self.decoder = DecoderVEGA(
            mask = mask.T,
            n_cat_list=cat_list,
            n_continuous_cov=n_continuous_cov,
            positive_decoder = self.positive_decoder,
            regularizer = self.regularizer,
            reg_kwargs = self.reg_kwargs,
            )
        
    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            z=z,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v)
        return outputs

    @auto_move_data
    def generative(self,
        z,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        px = self.decoder(decoder_input,
            batch_index,
            *categorical_input,
            y,)

        return dict(px=px)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):

        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        p = generative_outputs["px"]

        kld = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)
        rl = self.get_reconstruction_loss(p, x)
        loss = (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean()
        # Get quadratic penalty on decoder's weights in case of L2 constraints (eg. GelNet)
        l2_pen = self.decoder.quadratic_penalty()
        loss += l2_pen
        return LossRecorder(loss, rl, kld, l2_penalty=l2_pen)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        if n_samples>1:
            px = Normal(generative_outputs["px"], 1).sample().permute([1,2,0])
        else:
            px = Normal(generative_outputs["px"], 1).sample()
        return px.cpu().numpy()

    def get_reconstruction_loss(self, x, px) -> torch.Tensor:
        loss = ((x - px) ** 2).sum(dim=1)
        return loss


    def proximal_update(self):
        """
        Call proximal updates on the linear decoder of the VEGA module.
        """
        self.decoder.proximal_update()
        return

    def _positive_constraint(self):
        """
        Enforce positivity constraint on decoder's weights.
        """
        if self.positive_decoder:
            self.decoder._positive_weights()
        return
