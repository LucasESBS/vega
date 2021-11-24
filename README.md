[![DOI](https://zenodo.org/badge/326273034.svg)](https://zenodo.org/badge/latestdoi/326273034)
# VEGA: VAE Enhanced by Gene Annotations
### _A VAE for analyzing pathways, transcription factors, cell types in single-cell RNA-seq data_

## Introduction
VEGA is a VAE aimed at analyzing _a priori_ specified latent variables such as pathways. VEGA is implemented with pytorch.

## Getting started
VEGA needs 2 things to analyze your data:


* A single-cell dataset wrapped using the Scanpy package (Wolf et al. 2018)
* A GMT file specifying the gene module variables (GMVs) and gene membership, eg. from MSigDB

## Differential GMVs activity
VEGA provides a statistical framework for identifying differentially activated GMVs. See vega-reproducibility github for examples.

## Reproducing paper results
VEGA manuscript results can be reproduced using [the following code](https://github.com/LucasESBS/vega-reproducibility). Check tags for appropriate version of VEGA for reproducing results.

## Manuscript
VEGA preprint can be found [here](https://www.biorxiv.org/content/10.1101/2020.12.17.423310v1.abstract).
