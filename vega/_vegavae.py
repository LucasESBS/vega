import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from vega._base_components import DecoderVEGA


class VEGAVAE(BaseModuleClass):
    """
    VEGA: VAE Enhanced by Gene Annotations model.

    Parameters
    ----------

    """
    def __init__(
        self,
        mask: np.ndarray,
        n_hidden: int = 800, 
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        z_dropout: float = 0.,
        positive_decoder: bool = False,
        regularizer: Literal["mask", "l1", "gelnet"],
        reg_kwargs: dict = None,
        log_variational: bool = False,
        latent_distribution: str = 'normal',
        use_batch_norm: Literal["encoder","decoder","none","both"] = "encoder",
        use_layer_norm: Literal["encoder","decoder","none","both"] = "none",
        kl_weight: float = 5e-5,
    ):
        super().__init__()
        self.mask
        self.n_input = mask.shape[0]
        self.n_gmvs = mask.shape[1]
        self.n_latent = mask.shape[1]
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

        # Encoder for GMV activity
        self.z_encoder = Encoder(
            self.n_input,
            self.n_gmvs,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=torch.nn.ReLU,
            )

        # Add dropout to mean and var of z_encoder
        self.z_encoder.mean_encoder = torch.nn.Sequential(self.z_encoder.mean_encoder, torch.nn.Dropout(p=z_dropout))
        self.z_encoder.var_encoder = torch.nn.Sequential(self.z_encoder.var_encoder, torch.nn.Dropout(p=z_dropout))

        # Decoder Gaussian
        self.decoder = DecoderVEGA(
            mask = self.mask.T,
            positive_decoder = self.positive_decoder
            regularizer = self.regularizer,
            reg_kwargs = self.reg_kwargs,
            )
        
    
