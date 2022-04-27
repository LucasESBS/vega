import inspect
import logging
import warnings
from functools import partial
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi._compat import Literal
from _utils import _da_core

logger = logging.getLogger(__name__)


class LatentMixin:
    """ Methods for analysis of interpretable latent space. """

    def differential_activity(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2:Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla","change"] = "change",
        delta: float = 2.,
        change_fn: Literal["diff","lfc"] = "diff",
        batch_size: Optional[int] = None,
        fdr_target: float = 0.1,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Method for measuring differential activities of Gene Module Variables (GMVs).
        Extends differential expression analysis strategy from scvi-tools to interpretable 
        latent variables.
        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_. For `"change"`,
        the function is simply the difference between the GMVs activations.

        Parameters
        ----------
        Returns
        -------
        Differential activity DataFrame.
        """
        adata = self._validate_anndata(adata)
        # Not great syntax but leave for now 
        assert "_vega" in adata.uns.keys(), "Input AnnData not setup with VEGA. Please run `VEGA.setup_anndata()`"
        assert not (self.module.latent_distribution=="normal" and change_fn=="lfc"), "Can't use LFC with normal latent distribution."

        # Get GMV names from VEGA field
        col_names = adata.uns["_vega"]["gmv_names"]
        # Define function -- here get latent space
        model_fn = partial(
            self.get_latent_representation,
            batch_size=batch_size,
            give_mean=True if self.module.latent_distribution=='normal' else False
        )

        result = _da_core(
            self.get_anndata_manager(adata, required=True),
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
            fdr_target,
            silent,
            **kwargs,
        )

        return result
