import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.module import VAE
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.nn import Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import Callable, Iterable, Optional

from vega.utils import DecoderVEGACount


class VEGAVAECount(BaseModuleClass):
    """
    VEGA: VAE Enhanced by Gene Annotations model.
    Implementation to model count data.

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
        dispersion: str = "gene",
        gene_likelihood: str = "nb",
        log_variational: bool = False,
        latent_distribution: str = 'normal',
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder","decoder","none","both"] = "encoder",
        use_layer_norm: Literal["encoder","none"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        kl_weight: float = 5e-5,
        bias: bool = True,
    ):
        super().__init__()
        
        n_input = mask.shape[0]
        n_gmvs = mask.shape[1]

        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent = n_gmvs
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.kl_weight = kl_weight
        self.z_dropout = z_dropout
        self.positive_decoder = positive_decoder
        self.regularizer = regularizer
        self.reg_kwargs = reg_kwargs
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
      
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        # Layer norm always false for decoder
         
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            raise ValueError("dispersion cannot be 'gene-cell' in VEGA")
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label'], but input was "
                "{}.format(self.dispersion)"
            )
 
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
            var_activation=var_activation,
            )
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )

        # Add dropout to mean and var of z_encoder
        self.z_encoder.mean_encoder = torch.nn.Sequential(self.z_encoder.mean_encoder, torch.nn.Dropout(p=z_dropout))
        self.z_encoder.var_encoder = torch.nn.Sequential(self.z_encoder.var_encoder, torch.nn.Dropout(p=z_dropout))

        # Decoder Poisson/NB/ZINB
        self.decoder = DecoderVEGACount(
            mask = mask.T,
            n_cat_list=cat_list,
            n_continuous_cov=n_continuous_cov,
            positive_decoder = self.positive_decoder,
            regularizer = self.regularizer,
            reg_kwargs = self.reg_kwargs,
            use_batch_norm = use_batch_norm_decoder,
            use_layer_norm=False,
            bias = bias,
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
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        input_dict = dict(
            z=z,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
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

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)

        if not self.use_observed_lib_size:
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                Normal(ql_m, ql_v.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)
        # Get quadratic penalties from decoder regularizer(s)
        l2_pen = self.decoder.quadratic_penalty()
        loss += l2_pen        

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global, l2_penalty=l2_pen)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
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

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "poisson":
            l_train = px_rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
        else:
            raise ValueError(
                "{} reconstruction error not handled right now".format(
                    self.module.gene_likelihood
                )
            )
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            log_prob_sum += p_z + p_x_zl

            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            log_prob_sum -= q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )

                ql_m = inference_outputs["ql_m"]
                ql_v = inference_outputs["ql_v"]
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl

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
