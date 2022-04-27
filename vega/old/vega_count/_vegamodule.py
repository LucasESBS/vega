import numpy as np
import torch
from scvi.module import VAE
from typing import Iterable, Optional
#from scvi import _CONSTANTS
#from scvi.distributions import ZeroInflatedNegativeBinomial

#from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
#from torch.distributions import Normal
#from torch.distributions import kl_divergence as kl
from vega.layers import DecoderVEGACount

torch.backends.cudnn.benchmark = True


class SparseVAE(VAE):
    """
    VEGA: Variational auto-encoder Enhanced by Gene Annotations [Seninge20]_.

    VEGA is a VAE-based model for inference of Gene Module Variables (GMVs) activities. It uses
    a binary matrix initialized from a .gmt file to mask connection in a linear decoder
    in order to provide interpretability over each latent variable of the model,
    providing an easy way to project data into the interpretable latent space and perform
    differential testing.

    Parameters
    ----------
    gmv_mask
        Mask for Gene Module annotations. Dimensions: n_genes x n_modules
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        n_input: int,
        gmv_mask: np.ndarray,
        n_batch: int = 0,
        n_hidden: int = 600,
        n_layers: int = 3,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        z_dropout: float = 0,
        gene_likelihood: str = "zinb",
        encode_covariates: bool = False
    ):
        super().__init__(n_input=n_input)
        self.mask = gmv_mask
        self.n_genes = gmv_mask.shape[0]
        self.n_gmvs = gmv_mask.shape[1]
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        n_input_encoder = self.n_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.latent_distribution = "normal"
        # Model parameters (gene dispersions)
        self.px_r = torch.nn.Parameter(torch.randn(self.n_genes))
        # GMV activity encoder
        self.z_encoder = Encoder(
                                n_input_encoder,
                                self.n_gmvs,
                                n_cat_list = [n_batch] if encode_covariates else None,
                                n_layers=n_layers,
                                n_hidden=n_hidden,
                                dropout_rate=dropout_rate,
                                use_batch_norm=True
                                )
        # Add dropout to mean and var of z_encoder
        self.z_encoder.mean_encoder = torch.nn.Sequential(self.z_encoder.mean_encoder, torch.nn.Dropout(p=z_dropout))
        self.z_encoder.var_encoder = torch.nn.Sequential(self.z_encoder.var_encoder, torch.nn.Dropout(p=z_dropout))
        # Scaling factor encoder
        self.l_encoder = Encoder(
                                n_input_encoder,
                                1,
                                n_cat_list=[n_batch] if encode_covariates else None,
                                n_layers=1,
                                n_hidden=n_hidden,
                                dropout_rate=dropout_rate,
                                use_batch_norm=True
                                )
        # Sparse decoder to decode GMV activities
        # TO DO: Add continuous covariates to dim
        self.decoder = DecoderVEGACount(
                                self.mask.T,
                                n_cat_list = cat_list,
                                n_continuous_cov = n_continuous_cov,
                                use_batch_norm = False,
                                use_layer_norm = False,
                                bias = False
                                )

