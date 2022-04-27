import inspect
import logging
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi._types import Number
from scvi.model.base._differential import pairs_sampler, estimate_delta, credible_intervals, describe_continuous_distrib


logger = logging.getLogger(__name__)


class DifferentialActivityComputation:
    """
    Modified DifferentialComputation class from scvi to accomodates specificities about
    VEGA differential activity testing procedure.

    Parameters
    ----------
    model_fn
        Function in model API to get values from.
    adata_manager
        AnnDataManager created by :meth:`~scvi.model.SCVI.setup_anndata`. 
    """
   
    def __init__(self, model_fn, adata_manager):
        self.adata_manager = adata_manager
        self.adata = adata_manager.adata
        self.model_fn = model_fn

    def get_bayes_factors(
        self,
        idx1: Union[List[bool], np.ndarray],
        idx2: Union[List[bool], np.ndarray],
        mode: Literal["vanilla", "change"] = "vanilla",
        n_samples: int = 5000,
        use_permutation: bool = False,
        m_permutation: int = 10000,
        change_fn: Optional[Union[str, Callable]] = None,
        m1_domain_fn: Optional[Callable] = None,
        delta: Optional[float] = 2.,
        cred_interval_lvls: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Make docs later.
        """
        eps = 1e-8
        z_group1 = self.z_sampler(
                        selection=idx1,
                        n_samples=n_samples,
                        )
        z_group2 = self.z_sampler(
                        selection=idx2,
                        n_samples=n_samples,
                        )
        z_mean1 = z_group1.mean(axis=0)
        z_mean2 = z_group2.mean(axis=0)

        # Create pairs
        z_1, z_2 = pairs_sampler(
                z_group1,
                z_group2,
                use_permutation=use_permutation,
                m_permutation=m_permutation,
            )
        
        # Perform DA test
        if mode == "vanilla":
            logger.debug("Differential activity using vanilla mode")
            proba_m1 = np.mean(z_1 > z_2, 0)
            proba_m2 = 1.0 - proba_m1
            res = dict(
                proba_m1=proba_m1,
                proba_m2=proba_m2,
                bayes_factor=np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                z1=z_mean1,
                z2=z_mean2,
            )

        elif mode == "change":
            logger.debug("Differential activity using change mode")

            # step 1: Construct the change function
            def lfc(x, y):
                return np.log2(x) - np.log2(y)

            def diff(x, y):
                return x - y

            if change_fn == "diff" or change_fn is None:
                change_name = "diff"
                change_fn = diff

            elif change_fn == "lfc":
                change_name = "lfc"
                change_fn = lfc
                
            elif not callable(change_fn):
                raise ValueError("'change_fn' attribute not understood")

            else:
                change_name = "custom"
            # step2: Construct the DA area function
            if m1_domain_fn is None:

                def m1_domain_fn(samples):
                    delta_ = (
                        delta
                        if delta is not None
                        else estimate_delta(lfc_means=samples.mean(0))
                    )
                    logger.debug("Using delta ~ {:.2f}".format(delta_))
                    return np.abs(samples) >= delta_

            change_fn_specs = inspect.getfullargspec(change_fn)
            domain_fn_specs = inspect.getfullargspec(m1_domain_fn)
            if (len(change_fn_specs.args) != 2) | (len(domain_fn_specs.args) != 1):
                raise ValueError(
                    "change_fn should take exactly two parameters as inputs; m1_domain_fn one parameter."
                )
            try:
                change_distribution = change_fn(z_1, z_2)
                is_de = m1_domain_fn(change_distribution)
                delta_ = (
                    estimate_delta(lfc_means=change_distribution.mean(0))
                    if delta is None
                    else delta
                )
            except TypeError:
                raise TypeError(
                    "change_fn or m1_domain_fn have has wrong properties."
                    "Please ensure that these functions have the right signatures and"
                    "outputs and that they can process numpy arrays"
                )
            proba_m1 = np.mean(is_de, 0)
            change_distribution_props = describe_continuous_distrib(
                samples=change_distribution,
                credible_intervals_levels=cred_interval_lvls,
            )
            change_distribution_props = {
                change_name+ "_" + key: val for (key, val) in change_distribution_props.items()
            }

            res = dict(
                proba_da=proba_m1,
                proba_not_da=1.0 - proba_m1,
                bayes_factor=np.log(proba_m1 + eps) - np.log(1.0 - proba_m1 + eps),
                z1=z_mean1,
                z2=z_mean2,
                delta=delta_,
                **change_distribution_props,
            )
        else:
            raise NotImplementedError("Mode {mode} not recognized".format(mode=mode))

        return res

    @torch.no_grad()
    def z_sampler(
        self,
        selection: Union[List[bool], np.ndarray],
        n_samples: Optional[int] = 5000,
        n_samples_per_cell: Optional[int] = None,
        give_mean: Optional[bool] = False,
    ) -> dict:
        """
        Samples the variational posterior distribution to get GMV activities.
        Modified from scvi-tools `scale sampler`.
        
        Parameters
        ----------
        selection
            Mask or list of cell ids to select
        n_samples
            Number of samples in total per batch (fill either `n_samples_total`
            or `n_samples_per_cell`)
        n_samples_per_cell
            Number of time we sample from each observation per batch
            (fill either `n_samples_total` or `n_samples_per_cell`)
        give_mean
            Return mean of values
        
        Returns
        -------
        type
            np.ndarray of size (n_samples, n_gmvs) with GMV activities
        """
        # Get overall number of desired samples -- latent space is already batch corrected
        if n_samples is None and n_samples_per_cell is None:
            n_samples = 5000
        elif n_samples_per_cell is not None and n_samples is None:
            n_samples = n_samples_per_cell * len(selection)
        if (n_samples_per_cell is not None) and (n_samples is not None):
            warnings.warn(
                "n_samples and n_samples_per_cell were provided. Ignoring n_samples_per_cell"
            )
        if n_samples == 0:
            warnings.warn(
                "very small sample size, please consider increasing `n_samples`"
            )
            n_samples = 2

        # Selection of desired cells for sampling
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        selection = np.asarray(selection)
        if selection.dtype is np.dtype("bool"):
            if len(selection) < self.adata.shape[0]:
                raise ValueError("Mask must be same length as adata.")
            selection = np.asarray(np.where(selection)[0].ravel())
        # Up or down sample, then get latent
        idx_selected = np.arange(self.adata.shape[0])[selection]
        idx_selected = np.random.choice(idx_selected, n_samples)
        z_values =  self.model_fn(
                    self.adata,
                    indices=idx_selected,
                    )
        if give_mean:
            z_values = z_values.mean(0)
        return z_values


     
