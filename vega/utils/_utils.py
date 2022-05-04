from collections import OrderedDict
from collections.abc import Iterable as IterableClass
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from anndata import AnnData
import warnings
from sklearn.mixture import GaussianMixture
from typing import Union
from ._differential import DifferentialActivityComputation
from scvi.utils import track

def create_mask(adata: AnnData,
                gmt_paths: Union[str,list] = None,
                add_nodes: int = 1,
                min_genes: int = 0,
                max_genes: int = 1000,
                copy: bool = False):
    """ 
    Initialize mask M for GMV from one or multiple .gmt files.

    Parameters
    ----------
        adata
            Scanpy single-cell object.
        gmt_paths
            One or several paths to .gmt files.
        add_nodes
            Additional latent nodes for capturing additional variance.
        min_genes
            Minimum number of genes per GMV.
        max_genes
            Maximum number of genes per GMV.
        copy
            Whether to return a copy of the updated Anndata object.

    Returns
    -------
        adata
            Scanpy single-cell object.
    """
    if copy:
        adata = adata.copy()
    # Check if mask already exists
    if 'mask' in adata.uns['_vega'].keys():
        raise ValueError(
            " Mask already existing in Anndata object. Re-run setup_anndata() if you wish to erase previous mask information."
        )
    dict_gmv = OrderedDict()
    # Check if path is a string
    if type(gmt_paths) == str:
        gmt_paths = [gmt_paths]
    for f in gmt_paths:
        d_f = _read_gmt(f, sep='\t', min_g=min_genes, max_g=max_genes)
        # Add to final dictionary
        dict_gmv.update(d_f)

    # Create mask
    mask = _make_gmv_mask(feature_list=adata.var.index.tolist(), dict_gmv=dict_gmv, add_nodes=add_nodes)

    adata.uns['_vega']['mask'] = mask
    adata.uns['_vega']['gmv_names'] = list(dict_gmv.keys()) + ['UNANNOTATED_'+str(k) for k in range(add_nodes)]
    adata.uns['_vega']['add_nodes'] = add_nodes

    if copy:
        return adata
        

def _make_gmv_mask(feature_list, dict_gmv, add_nodes):
    """ 
    Creates a mask of shape [genes,GMVs] where (i,j) = 1 if gene i is in GMV j, 0 else.
    Note: dict_gmv should be an Ordered dict so that the ordering can be later interpreted.

    Parameters
    ----------
        feature_list
            List of genes in single-cell dataset.
        dict_gmv
            Dictionary of gene_module:genes.
        add_nodes
            Number of additional, fully connected nodes.

    Returns
    -------
        p_mask
            Gene module mask
    """
    assert type(dict_gmv) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_gmv)))
    for j, k in enumerate(dict_gmv.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_gmv[k]:
                p_mask[i,j] = 1.
    # Add unannotated nodes
    n = add_nodes
    vec = np.ones((p_mask.shape[0], n))
    p_mask = np.hstack((p_mask, vec))
    return p_mask

def _dict_to_gmt(dict_obj, path_gmt, sep='\t', second_col=True):
    """ 
    Write dictionary to gmt format.

    Parameters
    ----------
        dict_obj
            Dictionary with gene_module:[members]
        path_gmt
            Path to save gmt file
        sep
            Separator to use when writing file
        second_col
            Whether to duplicate the first column        
    """
    with open(path_gmt, 'w') as f:
        for k,v in dict_obj.items():
            if second_col:
                to_write = sep.join([k,'SECOND_COL'] + v)+'\n'
            else:
                to_write = sep.join([k] + v) + '\n'
            f.write(to_write)
    return
         

def _read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.
    min_g and max_g are optional gene set size filters.
    
    Parameters
    ----------
        fname
            Path to gmt file
        sep
            Separator used to read gmt file.
        min_g
            Minimum of gene members in gene module
        max_g
            Maximum of gene members in gene module
    Returns
    -------
        dict_gmv
            Dictionary of gene_module:genes
    """
    dict_gmv = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_gmv[val[0]] = val[2:]
    return dict_gmv

def preprocess_anndata(adata: AnnData,
                        n_top_genes:int = 5000,
                        copy: bool = False):
    """
    Simple (default) Scanpy preprocessing function before autoencoders.

    Parameters
    ----------
        adata
            Scanpy single-cell object
        n_top_genes
            Number of highly variable genes to retain
        copy
            Return a copy or in place
    Returns
    -------
        adata
            Preprocessed Anndata object
    """
    if copy:
        adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    if copy:
        return adata


def _da_core(
    adata_manager,
    model_fn,
    groupby,
    group1,
    group2,
    idx1,
    idx2,
    col_names,
    mode,
    delta,
    change_fn,
    fdr,
    silent,
    **kwargs,
):
    """Internal function for differential activity interface."""
    adata = adata_manager.adata
    if group1 is None and idx1 is None:
        group1 = adata.obs[groupby].astype("category").cat.categories.tolist()
        if len(group1) == 1:
            raise ValueError(
                "Only a single group in the data. Can't run DA on a single group."
            )

    if not isinstance(group1, IterableClass) or isinstance(group1, str):
        group1 = [group1]

    # make a temp obs key using indices
    temp_key = None
    if idx1 is not None:
        obs_col, group1, group2 = _prepare_obs(idx1, idx2, adata)
        temp_key = "_vega_temp_da"
        adata.obs[temp_key] = obs_col
        groupby = temp_key

    df_results = []
    dc = DifferentialActivityComputation(model_fn, adata_manager)
    for g1 in track(
        group1,
        description="DA...",
        disable=silent,
    ):
        cell_idx1 = (adata.obs[groupby] == g1).to_numpy().ravel()
        if group2 is None:
            cell_idx2 = ~cell_idx1
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()

        all_info = dc.get_bayes_factors(
            cell_idx1,
            cell_idx2,
            mode=mode,
            delta=delta,
            **kwargs,
        )

        res = pd.DataFrame(all_info, index=col_names)
        sort_key = "proba_da" if mode == "change" else "bayes_factor"
        res = res.sort_values(by=sort_key, ascending=False)
        if mode == "change":
            res["is_da_fdr_{}".format(fdr)] = _fdr_de_prediction(
                res["proba_da"], fdr=fdr
            )
        if idx1 is None:
            g2 = "Rest" if group2 is None else group2
            res["comparison"] = "{} vs {}".format(g1, g2)
            res["group1"] = g1
            res["group2"] = g2
        df_results.append(res)

    if temp_key is not None:
        del adata.obs[temp_key]

    result = pd.concat(df_results, axis=0)

    return result


def _estimate_delta(metric_means, min_threshold=1., coef=0.6):
    """
    Estimating delta from GMM. Taken from scvi-tools.
    """
    gmm = GaussianMixture(n_components=3)
    gmm.fit(metric_means[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = coef * np.abs(vals[[0, -1]]).mean()
    res = np.maximum(min_threshold, res)
    return res

def _fdr_de_prediction(posterior_probas: np.ndarray, fdr: float = 0.05):
    """
    Compute posterior expected FDR and tag features as DE.
    From scvi-tools.
    """
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    sorted_genes = np.argsort(-posterior_probas)
    sorted_pgs = posterior_probas[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    pred_de_genes = sorted_genes[:d]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return is_pred_de
