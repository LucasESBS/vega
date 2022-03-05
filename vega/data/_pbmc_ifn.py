import scanpy as sc

def _load_pbmc():
    """ 
    Reload PBMC dataset with IFN treatment (Kang et al.)
    """
    print('Loading demonstration data...')
    adata = sc.read('./train_pbmc.h5ad', backup_url='https://dl.dropbox.com/s/4obyq8rzd7sjn99/train_pbmc.h5ad?dl=1')
    return adata
