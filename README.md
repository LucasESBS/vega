# VEGA: VAE Enhanced by Gene Annotations
### _A VAE for analyzing pathways, transcription factors, cell types in single-cell RNA-seq data_

## Introduction
VEGA is a VAE aimed at analyzing _a priori_ specified latent variables such as pathways. VEGA is implemented with pytorch.

## Installation
To install VEGA, clone this repository:
```bash
git clone https://github.com/LucasESBS/vega
```
Then run in the VEGA repository:
```bash
python setup.py install
```

## Getting started
VEGA needs 2 things to analyze your data:


* A single-cell dataset wrapped using the Scanpy package (Wolf et al. 2018)
* A GMT file specifying the gene module variables (GMVs) and gene membership, eg. from MSigDB

We recommend that the Scanpy Anndata object is preprocessed before passed to VEGA:
```python
import scanpy as sc
adata = sc.read(path)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
```
We also recommend using a subset of highly variable genes (5000-7000). See the [Scanpy documentation](https://scanpy.readthedocs.io/en/stable/index.html) for more information on preprocessing.

## Reproducing paper results
VEGA manuscript results can be reproduced using [the following code](https://github.com/LucasESBS/vega-reproducibility).

## Manuscript
VEGA preprint can be found [here](https://www.biorxiv.org/content/10.1101/2020.12.17.423310v1.abstract).
