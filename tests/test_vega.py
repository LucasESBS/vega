import os
import vega
import numpy as np
import pandas as pd
from anndata import AnnData


def generate_synthetic(n_cells=1000, n_genes=100):
    # generate fake data and return anndata
    # to match test_gmt.gmt gene names
    assert n_genes > 20
    x = np.random.rand(n_cells, n_genes)
    adata = AnnData(X=x, obs=pd.DataFrame({'ct':['fake']*n_cells}), var=pd.DataFrame({'gene_name':[str(i) for i in range(n_genes)]}))
    return adata

def test_base_vega():
    """
    Test VEGA basic functionalities
    """
    adata = generate_synthetic()
    # Test setup
    vega.utils.setup_anndata(adata)
    assert '_vega' in adata.uns.keys()
    assert '_scvi' in adata.uns.keys()
    # Test init, rep and train
    model = vega.VEGA(adata,
                  gmt_paths='test_gmt.gmt',
                  add_nodes=1,
                  positive_decoder=True)
    print(model)
    model.train_vega(n_epochs=1)
    # Test latent
    z = model.to_latent()
    # Test save and load
    model.save('./test_save/', save_adata=True, save_history=True)
    model = vega.VEGA.load('./test_save/')
    assert model.is_trained_
    return

if __name__=="__main__":
    test_base_vega()
    print('Test is successful')

