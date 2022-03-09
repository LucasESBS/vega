#!/usr/bin/env python3

# Module for VEGA
import os
import inspect
import pickle
import collections
import torch
from typing import Union
#import logging
import numpy as np
import pandas as pd
from anndata import AnnData
import torch.nn.functional as F
from torch import nn, optim
from vega.utils import *
from vega.utils import _anndata_loader, _anndata_splitter, _scvi_loader, _estimate_delta, _fdr_de_prediction
from vega.learning_utils import *
import scanpy as sc
from scipy import sparse

# SCVI and VEGA layers
from scvi.dataloaders import AnnDataLoader
from scvi.nn import FCLayers
from scvi import _CONSTANTS
from vega.layers import SparseLayer, DecoderVEGA

class VEGA(torch.nn.Module):
    def __init__(self, adata: AnnData,
                gmt_paths: Union[list,str] =None, 
                add_nodes: int = 1,
                min_genes: int = 0,
                max_genes: int =5000,
                positive_decoder: bool = True,
                encode_covariates: bool = False,
                regularizer: str = 'mask',
                reg_kwargs: dict = None,
                **kwargs):
        """ 
        Constructor for class VEGA (VAE Enhanced by Gene Annotations).

        Parameters
        ----------
        adata
            scanpy single-cell object. Please run setup_anndata() before passing to VEGA.
        gmt_paths
            one or more paths to .gmt files for GMVs initialization.
        add_nodes
            additional fully-connected nodes in the mask.
        min_genes
            minimum gene size for GMVs.
        max_genes
            maximum gene size for GMVs.
        positive_decoder
            whether to constrain decoder to positive weights
        encode_covariates
            whether to encode covariates along gene expression
        regularizer
            which regularization strategy to use (l1, gelnet, mask). Default: mask.
        reg_kwargs
            parameters for regularizer.
        **kwargs
            use_cuda
                using CPU (False) or CUDA (True).
            beta
                weight for KL-divergence.
            dropout
                dropout rate in model
            z_dropout
                dropout rate for the latent space (for correlation).
        """
        super(VEGA, self).__init__()
        self.adata = adata
        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        # Check for setup and mask existence
        if '_vega' not in self.adata.uns.keys():
            raise ValueError('Please run vega.utils.setup_anndata(adata) before initializing VEGA.')
        if 'mask' not in self.adata.uns['_vega'].keys() and not gmt_paths:
            raise ValueError('No existing mask found in Anndata object and no .gmt files passed to VEGA. Please provide .gmt file paths to initialize a new mask or use an Anndata object used for training of a previous VEGA model.')
        elif gmt_paths:
            create_mask(self.adata, gmt_paths, add_nodes, self.min_genes_, self.max_genes_)
            
        self.gmv_mask = adata.uns['_vega']['mask'] 
        self.n_gmvs = self.gmv_mask.shape[1]
        self.n_genes = self.gmv_mask.shape[0]
        self.use_cuda = kwargs.get('use_cuda', False)
        self.beta_ = kwargs.get('beta', 0.00005)
        self.dropout_ = kwargs.get('dropout', 0.1)
        self.z_dropout_ = kwargs.get('z_dropout', 0.3)
        self.pos_dec_ = positive_decoder
        self.regularizer_ = regularizer
        self.encode_covariates = encode_covariates
        self.epoch_history = {}
        # Categorical covariates
        n_cats_per_cov = (adata.uns['_scvi']['extra_categoricals']['n_cats_per_key']
                            if 'extra_categoricals' in adata.uns['_scvi'] else None)
        n_batch = adata.uns['_scvi']['summary_stats']['n_batch']
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # Model architecture
        self.encoder = FCLayers(n_in=self.n_genes,
                n_out=800,
                n_cat_list=cat_list if encode_covariates else None,
                n_layers=2,
                n_hidden=800,
                dropout_rate=self.dropout_)
        self.mean = nn.Sequential(nn.Linear(800, self.n_gmvs), 
                                    nn.Dropout(self.z_dropout_))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_gmvs), 
                                    nn.Dropout(self.z_dropout_))
        #self.decoder = SparseLayer(self.gmv_mask.T,
                                #n_cat_list=cat_list,
                                #use_batch_norm=False,
                                #use_layer_norm=False,
                                #bias=True,
                                #dropout_rate=0)
        # Setup decoder
        self.reg_kwargs = reg_kwargs
        self.decoder = DecoderVEGA(mask = self.gmv_mask.T,
                                    n_cat_list = cat_list,
                                    regularizer = self.regularizer_,
                                    positive_decoder = self.pos_dec_,
                                    reg_kwargs = self.reg_kwargs)
        # Other hyperparams
        self.is_trained_ = kwargs.get('is_trained', False) 
        # Constraining decoder to positive weights or not
        if self.pos_dec_:
            print('Constraining decoder to positive weights', flush=True)
            #self.decoder.sparse_layer[0].reset_params_pos()
            #self.decoder.sparse_layer[0].weight.data *= self.decoder.sparse_layer[0].mask
            self.decoder._positive_weights()    

    def __repr__(self):
        att = "VEGA model with the following parameters: \nn_GMVs: {}, dropout_rate:{}, z_dropout:{}, beta:{}, positive_decoder:{}".format(self.n_gmvs, self.dropout_, self.z_dropout_, self.beta_, self.pos_dec_)
        stat = "Model is trained: {}".format(self.is_trained_)
        return '\n'.join([att, stat])

    def _get_gmv_names(self):
        if not self.adata:
            raise ValueError('No Anndata object found')
        else:
            return list(self.adata.uns['_vega']['gmv_names'])

    def save(self,
            path: str,
            save_adata: bool = False,
            save_history: bool = False,
            overwrite: bool = False,
            save_regularizer_kwargs: bool = True):
        """ 
        Save model parameters to input directory. Saving Anndata object and training history is optional.

        Parameters
        ----------
        path
            path to save directory
        save_adata
            whether to save the Anndata object in the save directory
        save_history
            whether to save the training history in the save directory
        save_regularizer_kwargs
            whether to save regularizer hyperparameters (lambda, penalty matrix...) in the save directory
        """
        attr = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attr = [a for a in attr if not (a[0].startswith("__") and a[0].endswith("__"))]
        attr_dict = {a[0][:-1]:a[1] for a in attr if a[0][-1]=='_'}
        # Save
        if not os.path.exists(path) or overwrite:
            os.makedirs(path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    path
                )
            )
        
        with open(os.path.join(path, 'vega_attr.pkl'), 'wb') as f:
            pickle.dump(attr_dict, f)
        torch.save(self.state_dict(), os.path.join(path, 'vega_params.pt'))
        if save_adata:
            self.adata.write(os.path.join(path, 'anndata.h5ad'))
        if save_history:
            with open(os.path.join(path, 'vega_history.pkl'), 'wb') as h:
                pickle.dump(self.epoch_history, h)
        if self.reg_kwargs and save_regularizer_kwargs:
            with open(os.path.join(path, 'regularizer_kwargs.pkl'), 'wb') as r:
                pickle.dump(self.reg_kwargs, r)
        print("Model files saved at {}".format(path))
        return
    
    @classmethod
    def load(cls,
            path: str,
            adata: AnnData = None,
            device: torch.device = torch.device('cpu'),
            reg_kwargs: dict = None):
        """
        Load model from directory. If adata=None, try to reload Anndata object from saved directory.

        Parameters
        ----------
        path 
            path to save directory
        adata
            scanpy single cell object
        device
            CPU or CUDA
        """
        # Reload model attributes
        with open(os.path.join(path, 'vega_attr.pkl'), 'rb') as f:
            attr = pickle.load(f)
        # Reload regularizer if possible
        if 'reg_kwargs' not in attr and attr['regularizer'] != 'mask':
            try:
                with open(os.path.join(path, 'regularizer_kwargs.pkl'), 'rb') as r:
                    attr['reg_kwargs'] = pickle.load(r)
            except:
                attr['reg_kwargs'] = reg_kwargs
        # Reload Anndata
        if not adata:
            try:
                adata = sc.read(os.path.join(path, 'anndata.h5ad'))
            except:
                FileNotFoundError('No Anndata object was passed or found in input directory.')
        model = cls(adata, **attr)
        # Reload history if possible
        try:
            with open(os.path.join(path, 'vega_history.pkl'), 'rb') as h:
                model.epoch_history = pickle.load(h)
        except:
            print('No epoch history file found. Loading model with blank training history.')
        # Reload model weights
        model.load_state_dict(torch.load(os.path.join(path, 'vega_params.pt'), map_location=device))
        if model.is_trained_ :
            model.eval()
        print("Model successfully loaded.")
        return model

    def _get_inference_input(self, tensors):
        """ Parse tensor dictionary. From SCVI [Lopez2018]_. """
        X = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(X=X, batch_index=batch_index, cat_covs=cat_covs)
        return input_dict

    def _get_generative_input(self, tensors, z):
        """ Parse tensor dictionary for generative model. From SCVI [Lopez2018]_. """
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        return dict(z=z, batch_index=batch_index, cat_covs=cat_covs)

    def encode(self, X, batch_index, cat_covs=None):
        """ 
        Encode data in latent space (Inference step).

        Parameters
        ----------
        X
            input data
        batch_index
            batch information for samples
        cat_covs
            categorical covariates

        Returns
        ------
        z
            data in latent space
        mu
            mean of variational posterior
        logvar
            log-variance of variational posterior
        """
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        y = self.encoder(X, batch_index, *categorical_input)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z, batch_index, cat_covs=None):
        """ 
        Decode data from latent space.
        
        Parameters
        ----------
        z
            data embedded in latent space
        batch_index
            batch information for samples
        cat_covs
            categorical covariates.

        Returns
        -------
        X_rec
            decoded data
        """
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        X_rec = self.decoder(z, batch_index, *categorical_input)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ 
        Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z.

        Parameters
        ----------
        mu
            mean of variational posterior
        logvar
            log-variance of variational posterior

        Returns
        -------
        eps
            sampled latent space
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = eps.to(torch.device('cuda'))
        eps = eps.mul_(std).add_(mu)
        return eps

    @torch.no_grad()
    def to_latent(self,
                adata: AnnData = None,
                indices: list = None,
                return_mean: bool = False):
        """
        Project data into latent space. Inspired by SCVI.
        
        Parameters
        ----------
        adata
            scanpy single-cell dataset
        indices
            indices of the subset of cells to be encoded
        return_mean
            whether to use the mean of the multivariate gaussian or samples
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        if not adata:
            adata = self.adata
        sc_dl = AnnDataLoader(adata, indices=indices, batch_size=128)
        latent = []
        for tensors in sc_dl:
            input_encode = self._get_inference_input(tensors)
            z, mu, logvar = self.encode(**input_encode)
            if return_mean:
                latent += [mu.cpu()]
            else:
                latent += [z.cpu()]
            
        return np.array(torch.cat(latent))

    @torch.no_grad()
    def generative(self,
                    adata: AnnData = None,
                    indices: list = None,
                    use_mean: bool = True):
        """
        Generate new samples from input data (encode-decode).

        Parameters
        ----------
        adata
            scanpy single-cell dataset
        indices
            indices of the subset of cells to be encoded
        use_mean
            whether to use the mean of the multivariate gaussian or samples
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        if not adata:
            adata = self.adata
        sc_dl = AnnDataLoader(adata, indices=indices, batch_size=128)
        samples = []
        for tensors in sc_dl:
            input_encode = self._get_inference_input(tensors)
            z, mu, logvar = self.encode(**input_encode)
            gen_input = mu if use_mean else z
            input_decode = self._get_generative_input(tensors, gen_input)
            x_rec = self.decode(**input_decode)
            samples += [x_rec.cpu()]
        return np.array(torch.cat(samples))
            
    def _average_latent(self, X, batch_index, cat_covs=None):
        """
        Computes the average data vector in the latent space.
        """
        z = self.to_latent(X, batch_index,cat_covs)
        mean_z = z.mean(0)
        return mean_z
    
    @torch.no_grad()
    def differential_activity(self,
                            groupby: str,
                            adata: AnnData = None,
                            group1: Union[str,list] = None,
                            group2: Union[str,list] = None,
                            mode: str = 'change',
                            delta: float = 2.,
                            fdr_target: float = 0.05,
                            **kwargs):
        """
        Bayesian differential activity procedures for GMVs.
        Similar to scVI [Lopez2018]_ Bayesian DGE but for latent variables.
        Differential results are saved in the adata object and returned as a DataFrame.
 
        Parameters
        ----------
        groupby
            anndata object field to group cells (eg. `"cell type"`)
        adata
            scanpy single-cell object. If None, use Anndata attribute of VEGA.
        group1
            reference group(s).
        group2
            outgroup(s).
        mode
            differential activity mode. If `"vanilla"`, uses [Lopez2018]_, if `"change"` uses [Boyeau2019]_.
        delta
            differential activity threshold for `"change"` mode.
        fdr_target
            minimum FDR to consider gene as DE.
        **kwargs
            optional arguments of the bayesian_differential method.
        
        Returns
        -------
        Differential activity results
            
        """
        # Check Anndata object
        if not adata and not self.adata:
            raise ValueError("No Anndata object passed to VEGA or differential activity function.")
        elif not adata:
            print("Using VEGA's adata attribute for differential analysis")
            adata = self.adata
        # Check for grouping
        if not group1:
            print("No reference group: running 1-vs-rest analysis for .obs[{}]".format(groupby))
            group1 = adata.obs[groupby].unique()
        if not isinstance(group1, collections.Iterable) or type(group1)==str:
            group1 = [group1]
        # Loop over groups
        diff_res = dict()
        df_res = []
        for g in group1:
            # get indices and compute values
            idx_g1 = adata.obs[groupby] == g
            name_g1 = str(g)
            if not group2:
                idx_g2 = ~idx_g1
                name_g2 = 'rest'
            else: 
                idx_g2 = adata.obs[groupby] == group2
                name_g2 = str(group2)
            res_g = self.bayesian_differential(adata,
                                                idx_g1,
                                                idx_g2,
                                                mode=mode,
                                                delta=delta,
                                                **kwargs)
            diff_res[name_g1+' vs.'+name_g2] = res_g
            # report results as df
            df = pd.DataFrame(res_g, index=adata.uns['_vega']['gmv_names'])
            sort_key = "p_da" if mode == "change" else "bayes_factor"
            df = df.sort_values(by=sort_key, ascending=False)
            if mode == 'change':
                df['is_da_fdr_{}'.format(fdr_target)] = _fdr_de_prediction(df['p_da'], fdr=fdr_target)
            # Add names to result df
            df['comparison'] = '{} vs. {}'.format(name_g1, name_g2)
            df['group1'] = name_g1
            df['group2'] = name_g2
            df_res.append(df)
        # Concatenate df results
        result = pd.concat(df_res, axis=0)
        # Put results in Anndata object
        adata.uns['_vega']['differential'] = diff_res
        return result
    
    @torch.no_grad()    
    def bayesian_differential(self,
                                adata: AnnData,
                                cell_idx1: list, 
                                cell_idx2: list, 
                                n_samples: int = 5000, 
                                use_permutations: bool = True, 
                                n_permutations: int = 5000,
                                mode: int = 'change',
                                delta: float = 2.,
                                alpha: float = 0.66,
                                random_seed: bool = False):
        """ 
        Run Bayesian differential expression in latent space.
        Returns Bayes factor of all factors.

        Parameters
        ----------
        adata
            anndata single-cell object.
        cell_idx1
            indices of group 1.
        cell_idx2
            indices of group 2.
        n_samples
            number of samples to draw from the latent space.
        use_permutations
            whether to use permutations when computing the double integral.
        n_permutations
            number of permutations for MC integral.
        mode
            differential activity test strategy. `"vanilla"` corresponds to [Lopez2018]_, `"change"` to [Boyeau2019]_.
        delta
            for mode `"change"`, the differential threshold to be used.
        random_seed
            seed for reproducibility.

        Returns
        -------
        res
            dictionary with results (Bayes Factor, Mean Absolute Difference)
        """
        #self.eval()
        # Set seed for reproducibility
        #print(mode, delta, alpha)
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        if mode not in ['vanilla', 'change']:
            raise ValueError('Differential mode not understood. Pick one of "vanilla", "change"')
        epsilon = 1e-12
        # Subset data
        #if sparse.issparse(adata.X):
            #adata1, adata2 = adata.X.A[cell_idx1,:], adata.X.A[cell_idx2,:]
        #else:
        adata1, adata2 = adata[cell_idx1,:], adata[cell_idx2,:]
        # Sample cell from each condition
        idx1 = np.random.choice(np.arange(len(adata1)), n_samples)
        idx2 = np.random.choice(np.arange(len(adata2)), n_samples)
        # To latent
        z1 = self.to_latent(adata1, indices=idx1, return_mean=False)
        z2 = self.to_latent(adata2, indices=idx2, return_mean=False)
        # Compare samples by using number of permutations - if 0, just pairwise comparison
        # This estimates the double integral in the posterior of the hypothesis
        if use_permutations:
            z1, z2 = self._scale_sampling(z1, z2, n_perm=n_permutations)
        if mode=='vanilla':
            p_h1 = np.mean(z1 > z2, axis=0)
            p_h2 = 1.0 - p_h1
            md = np.mean(z1 - z2, axis=0)
            bf = np.log(p_h1 + epsilon) - np.log(p_h2 + epsilon) 
            # Wrap results
            res = {'p_h1':p_h1,
                    'p_h2':p_h2,
                    'bayes_factor': bf,
                    'differential_metric':md}
        else:
            diffs = z1 - z2
            md = diffs.mean(0)
            if not delta:
                delta = _estimate_delta(md, min_thresh=1., coef=0.6)
            p_da = np.mean(np.abs(diffs) > delta, axis=0)
            is_da_alpha = (np.abs(md) > delta) & (p_da > alpha)
            res = {'p_da':p_da,
                    'p_not_da':1.-p_da,
                    'bayes_factor':np.log(p_da+epsilon) - np.log((1.-p_da)+epsilon),
                    'is_da_alpha_{}'.format(alpha):is_da_alpha,
                    'differential_metric':md,
                    'delta':delta
                    }
        return res

    @staticmethod
    def _scale_sampling(arr1, arr2, n_perm=1000):
        """
        Use permutation to better estimate double integral (create more pair comparisons)
        Inspired by scVI (Lopez et al., 2018)

        Parameters
        ----------
        arr1
            array with data of group 1
        arr2
            array with data of group 2
        n_perm
            number of permutations

        Returns
        -------
        scaled1
            samples for group 1
        scaled2
            samples for group 2
        """
        u, v = (np.random.choice(arr1.shape[0], size=n_perm), np.random.choice(arr2.shape[0], size=n_perm))
        scaled1 = arr1[u]
        scaled2 = arr2[v]
        return scaled1, scaled2

    def forward(self, tensors):
        """
        Forward pass through full network.
        
        Parameters
        ----------
        tensors
            input data
        
        Returns
        -------
        out_tensors
            dictionary of output tensors
        """
        input_encode = self._get_inference_input(tensors)
        z, mu, logvar = self.encode(**input_encode)
        input_decode = self._get_generative_input(tensors, z)
        X_rec = self.decode(**input_decode)
        return dict(x_rec=X_rec, mu=mu, logvar=logvar)

    def vae_loss(self, model_input, model_output):
        """ 
        Custom loss for beta-VAE
        
        Parameters
        ----------
        model_input
            dict with input values
        model_output
            dict with output values

        Returns
        -------
        loss value for current batch
        """
        # Parse values
        mu, logvar = model_output['mu'], model_output['logvar']
        y_pred, y_true = model_output['x_rec'], model_input[_CONSTANTS.X_KEY]
        # Get Loss
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta_*kld)

    def train_vega(self,
                    learning_rate: float = 1e-4,
                    n_epochs: int = 500,
                    train_size: float = 1.,
                    batch_size: int = 128,
                    shuffle: bool = True,
                    use_gpu: bool = False,
                    **kwargs):
        """ 
        Main method to train VEGA.

        Parameters
        ----------
        learning_rate
            learning rate
        n_epochs
            number of epochs to train model
        train_size
            a number between 0 and 1 to indicate the proportion of training data. Test size is set to 1-train_size
        batch_size
            number of samples per batch
        shuffle
            whether to shuffle samples or not
        use_gpu
            whether to use GPU
        **kwargs
            other keyword arguments of the _train_model() method, like the early stopping patience
        """
        train_patience=kwargs.get('train_patience', 10)
        test_patience=kwargs.get('test_patience', 10)
        # Create dataloader from Anndata
        #train_data, test_data = _anndata_splitter(self.adata, train_size=train_size)
        #train_loader = _anndata_loader(train_data, batch_size=batch_size, shuffle=shuffle)
        #test_loader = _anndata_loader(test_data, batch_size=batch_size, shuffle=shuffle) if test_data else False
        dev = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.to(dev)
        train_loader, test_loader = _scvi_loader(self.adata, train_size=train_size, batch_size=batch_size, use_gpu=use_gpu)
        # Call training method
        self.epoch_history = self._train_model(train_loader=train_loader,
                                            learning_rate=learning_rate,
                                            n_epochs=n_epochs,
                                            train_patience=train_patience,
                                            test_patience=test_patience,
                                            test_loader=test_loader,
                                            device=dev
                                            )
        # Set to eval mode
        self.eval()
        return

    def _train_model(self, train_loader, learning_rate, n_epochs, train_patience=10, test_patience=10, test_loader=False, device=torch.device('cpu')):
        """
        Training for VEGA.

        Parameters
        ----------
        train_loader
            loader with training data
        learning_rate
            learning rate for training
        n_epochs
            number of maximum epochs to train the model
        train_patience
            early stopping patience for training loss
        test_patience
            early stopping patience for test loss
        test_loader
            if available, loader with test data

        Returns
        -------
            epoch_hist (dict): Training history
        """
        epoch_hist = {}
        epoch_hist['train_loss'] = []
        epoch_hist['valid_loss'] = []
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode='train')
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode='valid')
        #clipper = WeightClipper(frequency=1)
        # Train
        for epoch in range(n_epochs):
            loss_value = 0
            self.train()
            for model_input in train_loader:
                # Send input to device
                model_input = {k:v.to(device) for k,v in model_input.items()}
                optimizer.zero_grad()
                model_output = self.forward(model_input)
                loss = self.vae_loss(model_input, model_output)
                # Regularization quadratic term if applicable
                loss += self.decoder.quadratic_penalty()
                loss_value += loss.item()
                loss.backward()
                optimizer.step()
                # Regularization non-smooth update if applicable
                self.decoder.proximal_update()
                if self.pos_dec_:
                    #self.decoder.sparse_layer[0].apply(clipper)
                    self.decoder._positive_weights()
            # Get epoch loss
            epoch_loss = loss_value / len(train_loader.indices)
            epoch_hist['train_loss'].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict = self._test_model(test_loader, device)
                test_loss = test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                valid_ES(test_loss)
                print('[Epoch %d] | loss: %.3f | test_loss: %.3f |'%(epoch+1, epoch_loss, test_loss), flush=True)
                if valid_ES.early_stop or train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
            else:
                print('[Epoch %d] | loss: %.3f |' % (epoch + 1, epoch_loss), flush=True)
                if train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
        self.is_trained_ = True
        return epoch_hist
    
    def _test_model(self, loader, device):
        """
        Test model on input loader.
        """
        test_dict = {}
        loss = 0
        loss_func = self.vae_loss
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = {k:v.to(device) for k,v in data.items()}
                model_output = self.forward(data)
                loss += loss_func(data, model_output).item()
                loss += self.decoder.quadratic_penalty().item()
        test_dict['loss'] = loss/len(loader.indices)
        return test_dict


