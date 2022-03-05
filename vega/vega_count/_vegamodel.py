import logging

from anndata import AnnData
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, RNASeqMixin, VAEMixin
from scvi._compat import Literal
from typing import Dict, Iterable, Optional, Sequence, Union
import torch
from ._vegamodule import SparseVAE
from vega.utils import *

logger = logging.getLogger(__name__)

class VegaSCVI(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    VEGA: VAE Enhanced by Gene Annotations [Seninge2021]_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    gmt_paths
        A single or list of paths to .GMT files with gene annotations for GMVs initialization.
    add_nodes
        Number of additional fully-connected decoder nodes (unannotated GMVs).
    min_genes
        Minimum gene size for GMVs.
    max_genes
        Maximum gene size for GMVs.
    positive_decoder
        Whether to constrain decoder to positive weights.
    n_hidden
        Number of nodes per hidden layer.
    n_layers
        Number of hidden layers used for encoder NN.
    gene_likelihood
        Likelihood function for the generative model.
    dropout_rate
        Dropout rate for neural networks.
    use_cuda
        Using GPU with CUDA
    """
    def __init__(
                self,
                adata: AnnData,
                gmt_paths: Literal = None,
                add_nodes: int = 1,
                min_genes: int = 0,
                max_genes: int = 5000,
                positive_decoder: bool = True, 
                n_hidden: int = 600,
                n_layers: int = 2,
                gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
                dropout_rate: float = 0.1,
                z_dropout: float = 0,
                use_cuda: bool = True,
                **model_kwargs):
        super(VegaSCVI, self).__init__(adata)
        # Get attributes
        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        # Check for setup and mask existence
        if '_vega' not in self.adata.uns.keys():
            print('Initializing `_vega` field in adata.uns', flush=True)
            self.adata.uns['_vega'] = {}
        if 'mask' not in self.adata.uns['_vega'].keys() and not gmt_paths:
            raise ValueError('No existing mask found in Anndata object and no .gmt files passed to VEGA. Please provide .gmt file paths to initialize a new mask or use an Anndata object used for training of a previous VEGA model.')
        elif 'mask' not in self.adata.uns['_vega'] and gmt_paths:
            create_mask(self.adata, gmt_paths, add_nodes, self.min_genes_, self.max_genes_)
        self.gmv_mask = self.adata.uns['_vega']['mask']
        self.n_gmvs = self.gmv_mask.shape[1]
        self.n_genes = self.gmv_mask.shape[0]
        n_cats_per_cov = (self.scvi_setup_dict_["extra_categoricals"]["n_cats_per_key"] 
                            if "extra_categoricals" in self.scvi_setup_dict_ else None)
        self.module = SparseVAE(
                                n_input=self.n_genes,
                                gmv_mask = self.gmv_mask,
                                n_batch = self.summary_stats["n_batch"],
                                n_continuous_cov = self.summary_stats["n_continuous_covs"],
                                n_cats_per_cov = n_cats_per_cov,
                                n_hidden = n_hidden,
                                n_layers = n_layers,
                                gene_likelihood = gene_likelihood,
                                dropout_rate = dropout_rate,
                                z_dropout = z_dropout,
                                encode_covariates = False,
                                **model_kwargs
                                )
        self.init_params_ = self._get_init_params(locals())
        
    #@property
    #def _trainer_class(self):
        #return UnsupervisedTrainer

    #@property
    #def _scvi_dl_class(self):
        #return ScviDataLoader

    def differential_activity(self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool]]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs):
            adata = self._validate_anndata(adata)
            col_names = adata.uns['_vega']['gmv_names']
            return result
