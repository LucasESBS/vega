from typing import Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model._utils import _init_library_size
from scvi.utils import setup_anndata_dsp
from scvi._compat import Literal

#from _vegavae import VEGAVAE
#from _utils import create_mask

from vega.utils import create_mask, LatentMixin, RegularizedTrainingMixin
from vega.module import VEGAVAECount



class VEGASCVI(VAEMixin, LatentMixin, RegularizedTrainingMixin, BaseModelClass):
    """
    Implementation of VEGA: VAE Enhanced by Gene Annotations using scvi-tools probabilistic API.
    This model implements a version of VEGA modelling raw count data rather than log-normalized, akin to SCVI modeling.
    """
    def __init__(
        self,
        adata: AnnData,
        gmt_paths: Union[list,str] =None,
        n_hidden: int = 800,
        n_layers: int = 2,
        add_nodes: int = 1,
        min_genes: int = 0,
        max_genes: int =5000,
        positive_decoder: bool = True,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        regularizer: str = 'mask',
        reg_kwargs: dict = None,
        dropout_rate: float = 0.2,
        z_dropout: float = 0.,
        **model_kwargs,
    ):
        super(VEGASCVI, self).__init__(adata)
        
        # get batch and covariates infos
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        # VEGA attributes -- old see if needed
        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        self.pos_dec_ = positive_decoder
        self.regularizer_ = regularizer
        self.reg_kwargs = reg_kwargs

        # Check for setup and mask existence
        if '_vega' not in self.adata.uns.keys():
            raise ValueError('Please run vega.utils.setup_anndata(adata) before initializing VEGA.')
        if 'mask' not in self.adata.uns['_vega'].keys() and not gmt_paths:
            raise ValueError('No existing mask found in Anndata object and no .gmt files passed to VEGA. Please provide .gmt file paths to initialize a new mask or use an Anndata object used for training of a previous VEGA model.')
        elif gmt_paths:
            create_mask(self.adata, gmt_paths, add_nodes, self.min_genes_, self.max_genes_)
            
       
        gmv_mask = self.adata.uns['_vega']['mask'] 
        n_gmvs = gmv_mask.shape[1]
        n_genes = gmv_mask.shape[0]
        
        # For now don't worry about regularizer, just assume its mask
        # TO DO: Add batch, labels, continuous and cat covs to VAE
        self.module = VEGAVAECount(
            mask=gmv_mask,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            positive_decoder=positive_decoder,
            regularizer=regularizer,
            reg_kwargs=reg_kwargs,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VEGASCVI model with following parameters: \nn_GMVs: {}, n_layers: {}, n_hidden: {}, dropout_rate: {}, z_dropout: {}, regularizer: {}, "
            "dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(n_gmvs, n_layers, n_hidden, dropout_rate, z_dropout, regularizer, dispersion, gene_likelihood, latent_distribution)

        self.init_params_ = self._get_init_params(locals())
        
    def _get_gmv_names(self):
        if not self.adata:
            raise ValueError('No Anndata object found')
        else:
            return list(self.adata.uns['_vega']['gmv_names'])

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)        
        # Create VEGA specific fields
        adata.uns['_vega'] = {}
        # Figure out a fix to add back version
        #adata.uns['_vega']['version'] = vega.__version__
