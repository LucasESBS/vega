<img src="docs/source/_static/logo.png" width="400">


[![DOI](https://zenodo.org/badge/326273034.svg)](https://zenodo.org/badge/latestdoi/326273034)
### _A VAE for analyzing pathways, transcription factors, cell types in single-cell RNA-seq data_

## Introduction

VEGA is a VAE aimed at analyzing _a priori_ specified latent variables such as pathways. VEGA is implemented with [pytorch](https://pytorch.org/), and using the [scanpy](https://scanpy.readthedocs.io/en/stable/) and [scvi-tools](https://scvi-tools.org/) ecosystem.

## Getting started

VEGA needs 2 things to analyze your data:

* A single-cell dataset wrapped using the Scanpy package
* A GMT file specifying the gene module variables (GMVs) and gene membership, eg. from MSigDB

## Installation
#### With pip
With pip, you can run
```
pip install vega
```

## Documentation and issues

- A [documentation](https://vega-documentation.readthedocs.io/en/latest/index.html) is available with API reference, installation guide and tutorials.
- Please consider submitting an [issue](https://github.com/LucasESBS/vega/issues) on github if you encounter a bug.

## Reproducing paper results

VEGA manuscript results can be reproduced using [the following code](https://github.com/LucasESBS/vega-reproducibility). Check tags for appropriate version of VEGA for reproducing results.

## Reference

If VEGA is useful to your research, please consider citing our [Nature Communications article](https://www.nature.com/articles/s41467-021-26017-0).

```
Seninge, L., Anastopoulos, I., Ding, H. et al. VEGA is an interpretable generative model for inferring biological network activity in single-cell transcriptomics. Nat Commun 12, 5684 (2021). https://doi.org/10.1038/s41467-021-26017-0
```
