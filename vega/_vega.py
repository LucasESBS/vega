from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from vega._vegavae import VEGAVAE
from vega.utils import *

class VEGA(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Implementation of VEGA: VAE Enhanced by Gene Annotations using scvi-tools probabilistic API.
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
        regularizer: str = 'mask',
        reg_kwargs: dict = None,
        **model_kwargs,
    ):
        super(VEGA, self).__init__(adata)
        
        # get batch and covariates infos
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        self.pos_dec_ = positive_decoder
        self.regularizer_ = regularizer
        # Setup decoder
        self.reg_kwargs = reg_kwargs
        # Check for setup and mask existence
        if '_vega' not in self.adata.uns.keys():
            raise ValueError('Please run vega.utils.setup_anndata(adata) before initializing VEGA.')
        if 'mask' not in self.adata.uns['_vega'].keys() and not gmt_paths:
            raise ValueError('No existing mask found in Anndata object and no .gmt files passed to VEGA. Please provide .gmt file paths to initialize a new mask or use an Anndata object used for training of a previous VEGA model.')
        elif gmt_paths:
            create_mask(self.adata, gmt_paths, add_nodes, self.min_genes_, self.max_genes_)
            
        self.gmv_mask = self.adata.uns['_vega']['mask'] 
        self.n_gmvs = self.gmv_mask.shape[1]
        self.n_genes = self.gmv_mask.shape[0]

        # For now don't worry about regularizer, just assume its mask
        # TO DO: Add batch, labels, continuous and cat covs to VAE
        self.module = VEGAVAE(
            mask=self.gmv_mask,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            positive_decoder=positive_decoder,
            regularizer=regularizer,
            reg_kwargs=reg_kwargs,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VEGA model with following parameters: \nn_hidden: {}, n_GMVs: {}, n_layers: {}, dropout_rate: {}"
        ).format(n_hidden, self.n_gmvs, n_layers, dropout_rate)

        self.init_params_ = self._get_init_params(locals())
        
    def _get_gmv_names(self):
        if not self.adata:
            raise ValueError('No Anndata object found')
        else:
            return list(self.adata.uns['_vega']['gmv_names'])
