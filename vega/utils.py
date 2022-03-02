from collections import OrderedDict
import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from scvi.data import setup_anndata as scvi_setup
from scvi.dataloaders import DataSplitter
import vega
import warnings
from sklearn.mixture import GaussianMixture

def setup_anndata(adata, batch_key=None, categorical_covariate_keys=None, copy=False):
    """
    Creates VEGA fields in input Anndata object for mask.
    Also creates SCVI field which will be used for batch and covariates.
    Args:
        adata (Anndata): Scanpy single-cell object.
        copy (bool): Whether to return a copy or change in place.
        batch_key (str): Observation to be used as batch.
        categorical_covariate_keys (list, str): Observation to use as covariate keys.
    Return:
        adata (Anndata): updated object if copy is True.
    """
    print('Running VEGA and SCVI setup...', flush=True)
    if copy:
        adata = adata.copy()

    if adata.is_view:
        raise ValueError(
            "Current adata object is a View. Please run `adata = adata.copy()` or use copy=True"
        )
    
    adata.uns['_vega'] = {}
    adata.uns['_vega']['version'] = vega.__version__
    # Use scvi setup to get batch keys and covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scvi_setup(adata, batch_key=batch_key, categorical_covariate_keys=categorical_covariate_keys)

    if copy:
        return adata

def create_mask(adata, gmt_paths=None, add_nodes=1, min_genes=0, max_genes=1000, copy=False):
    """ 
    Initialize mask M for GMV from one or multiple .gmt files.
    Args:
        adata (Anndata): Scanpy single-cell object.
        gmt_paths (str or list): one or several paths to .gmt files.
        add_nodes (int): Additional latent nodes for capturing additional variance.
        min_genes (int): Minimum number of genes per GMV.
        max_genes (int): Maximum number of genes per GMV.
        copy (bool): Whether to return a copy of the updated Anndata object.
    Return:
        adata (Anndata): Scanpy single-cell object.
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
    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_gmv (OrderedDict): Dictionary of gene_module:genes.
        add_nodes (int): Number of additional, fully connected nodes.
    Return:
        p_mask (np.array): Gene module mask
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
    Args:
        dict_obj (dict): Dictionary with gene_module:[members]
        path_gmt (str): Path to save gmt file
        sep (str): Separator to use when writing file
        second_col (bool): Whether to duplicate the first column        
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
    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module
        max_g (int): Maximum of gene members in gene module
    Return:
        dict_gmv (OrderedDict): Dictionary of gene_module:genes
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

def _anndata_loader(adata, batch_size, shuffle=False):
    """
    Load Anndata object into pytorch standard dataloader.
    Args:
        adata (AnnData): Scanpy Anndata object.
        batch_size (int): Cells per batch.
        shuffle (bool): Whether to shuffle data or not.
    Return:
        sc_dataloader (torch.DataLoader): Dataloader containing the data.
    """
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    data = torch.Tensor(data)
    sc_dataloader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return sc_dataloader

def _anndata_splitter(adata, train_size):
    """
    Splits Anndata object into a training and test set. Test proportion is 1-train_size.
    Args:
        adata (Anndata): Scanpy Anndata object.
        train_size (float): Proportion of whole dataset in training. Between 0 and 1.
    Returns:
        train_adata (Anndata): Training data subset.
        test_adata (Anndata): Test data subset.
    """
    assert train_size != 0
    n = len(adata)
    n_train = int(train_size*n)
    #n_test = n - n_train
    perm_idx = np.random.permutation(n)
    train_idx = perm_idx[:n_train]
    test_idx = perm_idx[n_train:]
    train_adata = adata.copy()[train_idx,:]
    if len(test_idx) != 0:
        test_adata = adata.copy()[test_idx,:]
    else:
        test_adata = False
    return train_adata, test_adata
    

def _scvi_loader(adata, train_size, batch_size, use_gpu=False):
    """
    SCVI splitter. Returs SCVI loader for train and test set.
    """
    data_splitter = DataSplitter(
            adata,
            train_size=train_size,
            validation_size=1.-train_size,
            batch_size=batch_size,
            use_gpu=use_gpu)
    train_dl, test_dl, _ = data_splitter()
    return train_dl, test_dl
    


def preprocess_anndata(adata, n_top_genes=5000, copy=False):
    """
    Simple (default) Scanpy preprocessing function before autoencoders.
    Args:
        adata (Anndata): Scanpy single-cell object.
        n_top_genes (int): Number of highly variable genes to retain.
        copy (bool): Return a copy or in place.
    Return:
        adata (Anndata): Preprocessed Anndata object.
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
