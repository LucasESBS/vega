#!/usr/bin/env python3

# Module for VEGA
import os
import inspect
import pickle
import collections
import torch
#import logging
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
#from vega.layers import CustomizedLinear
from vega.utils import *
from vega.utils import _anndata_loader, _anndata_splitter, _scvi_loader
from vega.learning_utils import *
import scanpy as sc
from scipy import sparse

# SCVI and VEGA layers
from scvi.dataloaders import AnnDataLoader
from scvi.nn import FCLayers
from scvi import _CONSTANTS
from vega.layers import SparseLayer, DecoderVEGA

class VEGA(torch.nn.Module):
    def __init__(self, adata,
                gmt_paths=None, 
                add_nodes=1,
                min_genes=0,
                max_genes=5000,
                positive_decoder=True,
                encode_covariates=False,
                regularizer = 'mask',
                reg_kwargs = None,
                **kwargs):
        """ 
        Constructor for class VEGA (VAE Enhanced by Gene Annotations).
        Args:
            adata (Anndata): Scanpy single-cell object. Please run setup_anndata() before passing to VEGA.
            gmt_paths (str or list): One or more paths to .gmt files for GMVs initialization.
            add_nodes (int): Additional fully-connected nodes in the mask.
            min_genes (int): Minimum gene size for GMVs.
            max_genes (int): Maximum gene size for GMVs.
            positive_decoder (bool): whether to constrain decoder to positive weights
            encode_covariates (bool): whether to encode covariates along gene expression
            regularizer (str): which regularization strategy to use (l1, gelnet, mask). Default: mask.
            reg_kwargs (dict): parameters for regularizer.
        kwargs:
            use_cuda (bool): using CPU (False) or CUDA (True).
            beta (float): weight for KL-divergence.
            dropout (float): dropout rate in model.
            z_dropout (float): dropout rate for the latent space (for correlation).
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
        self.beta_ = kwargs.get('beta', 0.0001)
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
        self.decoder = DecoderVEGA(mask = self.gmv_mask.T,
                                    n_cat_list = cat_list,
                                    regularizer = self.regularizer_,
                                    positive_decoder = self.pos_dec_,
                                    reg_kwargs = reg_kwargs)
        # Other hyperparams
        self.is_trained_ = kwargs.get('is_trained', False) 
        # Constraining decoder to positive weights or not
        if self.pos_dec_:
            print('Constraining decoder to positive weights', flush=True)
            #self.decoder.sparse_layer[0].reset_params_pos()
            #self.decoder.sparse_layer[0].weight.data *= self.decoder.sparse_layer[0].mask
            self.decoder._positive_weights()    

    def __repr__(self):
        att = "VEGA model with the following parameters: \nn_GMVs: {}, dropout_rate:{}, beta:{}, positive_decoder:{}".format(self.n_gmvs, self.dropout_, self.beta_, self.pos_dec_)
        stat = "Model is trained: {}".format(self.is_trained_)
        return '\n'.join([att, stat])

    def save(self, path, save_adata=False, save_history=False, overwrite=False):
        """ 
        Save model parameters to input directory. Saving Anndata object and training history is optional.
        Args:
            path (str): Path to save directory.
            save_adata (bool): Whether to save the Anndata object in the save directory.
            save_history (bool): Whether to save the training history in the save directory.
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
        print("Model files saved at {}".format(path))
        return
    
    @classmethod
    def load(cls, path, adata=None, device=torch.device('cpu')):
        """
        Load model from directory. If adata=None, try to reload Anndata object from saved directory.
        Args:
            path (str): Path to save directory.
            adata (Anndata): Scanpy single cell object. 
            device (torch.device): CPU or CUDA.
        """
        # Reload model attributes
        with open(os.path.join(path, 'vega_attr.pkl'), 'rb') as f:
            attr = pickle.load(f)
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
        """ Parse tensor dictionary. From SCVI [Lopez2018]. """
        X = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(X=X, batch_index=batch_index, cat_covs=cat_covs)
        return input_dict

    def _get_generative_input(self, tensors, z):
        """ Parse tensor dictionary for generative model. From SCVI [Lopez2018]. """
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        return dict(z=z, batch_index=batch_index, cat_covs=cat_covs)

    def encode(self, X, batch_index, cat_covs=None):
        """ 
        Encode data in latent space (Inference step).
        Args:
            X (torch.tensor): input data
            batch_index (torch.tensor): batch information for samples
            cat_covs (torch.tensor): categorical covariates.
        Return:
            z (torch.tensor): data in latent space
            mu (torch.tensor): mean of variational posterior
            logvar (torch.tensor): log-variance of variational posterior
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
        Args:
            z (torch.tensor): data embedded in latent space
            batch_index (torch.tensor): batch information for samples
            cat_covs (torch.tensor): categorical covariates.
        Return:
            X_rec (torch.tensor): decoded data
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
        Args:
            mu (torch.tensor): mean of variational posterior
            logvar (torch.tensor): log-variance of variational posterior
        Return:
            eps (torch.tensor): sampled latent space
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = eps.to(torch.device('cuda'))
        eps = eps.mul_(std).add_(mu)
        return eps

    @torch.no_grad()
    def to_latent(self, adata=None, indices=None, return_mean=False):
        """
        Project data into latent space. Inspired by SCVI.
        Args:
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
 
    def _average_latent(self, X, batch_index, cat_covs=None):
        """
        Computes the average data vector in the latent space.
        Return:
            mean_z (torch.tensor): mean latent vector
        """
        z = self.to_latent(X, batch_index,cat_covs)
        mean_z = z.mean(0)
        return mean_z
    
    @torch.no_grad()
    def differential_activity(self, groupby, adata=None, group1=None, group2=None, **kwargs):
        """
        Bayesian differential activity procedures for GMVs.
        Similar to scVI [Lopez 2018] Bayesian DGE but for latent variables.
        Differential results are saved in the adata object 
        Args:
            groupby (str): Anndata object field to group cells (eg. 'cell type')
            adata (Anndata): Scanpy single-cell object. If None, use Anndata attribute of VEGA.
            group1 (list or str): Reference group(s).
            group2 (list or str): outgroup(s).
            **kwargs: optional arguments of the bayesian_differential method.
        Returns
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
                                                **kwargs)
            diff_res[name_g1+' vs.'+name_g2] = res_g
        # Put results in Anndata object
        adata.uns['_vega']['differential'] = diff_res
        return
    
    @torch.no_grad()    
    def bayesian_differential(self, adata,
                                cell_idx1, 
                                cell_idx2, 
                                n_samples=2000, 
                                use_permutations=True, 
                                n_permutations=1000, 
                                random_seed=False):
        """ 
        Run Bayesian differential expression in latent space.
        Returns Bayes factor of all factors.
        Args:
            adata (Anndata): Anndata single-cell object.
            cell_idx1 (bool): Indices of group 1.
            cell_idx2 (bool): Indices of group 2.
            n_samples (int): Number of samples to draw from the latent space.
            use_permutations (bool): Whether to use permutations when computing the double integral.
            n_permutations (int): Number of permutations for MC integral.
            random_seed (int): Seed for reproducibility.
        Return:
            res (dict): Dictionary with results (Bayes Factor, Mean Absolute Difference)
        """
        #self.eval()
        # Set seed for reproducibility
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
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
        p_h1 = np.mean(z1 > z2, axis=0)
        p_h2 = 1.0 - p_h1
        mad = np.abs(np.mean(z1 - z2, axis=0))
        bf = np.log(p_h1 + epsilon) - np.log(p_h2 + epsilon) 
        # Wrap results
        res = {'p_h1':p_h1,
                'p_h2':p_h2,
                'bayes_factor': bf,
                'mad':mad}
        return res

    @staticmethod
    def _scale_sampling(arr1, arr2, n_perm=1000):
        """
        Use permutation to better estimate double integral (create more pair comparisons)
        Inspired by scVI (Lopez et al., 2018)
        Args:
            arr1 (np.array): array with data of group 1
            arr2 (np.array): array with data of group 2
            n_perm (int): number of permutations
        Return:
            scaled1 (np.array): samples for group 1
            scaled2 (np.array): samples for group 2
        """
        u, v = (np.random.choice(arr1.shape[0], size=n_perm), np.random.choice(arr2.shape[0], size=n_perm))
        scaled1 = arr1[u]
        scaled2 = arr2[v]
        return scaled1, scaled2

    def forward(self, tensors):
        """
        Forward pass through full network.
        Args:
            X (torch.tensor): input data
        Return:
            X_rec (torch.Tensor): reconstructed data
            mu (torch.tensor): mean of variational posterior
            logvar (torch.tensor): log-variance of variational posterior
        """
        input_encode = self._get_inference_input(tensors)
        z, mu, logvar = self.encode(**input_encode)
        input_decode = self._get_generative_input(tensors, z)
        X_rec = self.decode(**input_decode)
        return dict(x_rec=X_rec, mu=mu, logvar=logvar)

    def vae_loss(self, model_input, model_output):
        """ 
        Custom loss for VAE.
        Args:
            model_input (dict): dict with input values
            model_output (dict): dict with output values
        Return:
            loss value for current batch
        """
        # Parse values
        mu, logvar = model_output['mu'], model_output['logvar']
        y_pred, y_true = model_output['x_rec'], model_input[_CONSTANTS.X_KEY]
        # Get Loss
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta_*kld)

    def train_vega(self, learning_rate=1e-4, n_epochs=500, train_size=1., batch_size=128, shuffle=True, use_gpu=False, **kwargs):
        """ 
        Main method to train VEGA.
        Args:
            learning_rate (float): Learning rate.
            n_epochs (int): Number of epochs to train model.
            train_size (float): A number between 0 and 1 to indicate the proportion of training data. Test size is set to 1-train_size.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle samples or not.
            use_gpu (bool): Whether to use GPU
            **kwargs: Other keyword arguments of the _train_model() method, like the early stopping patience.
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
        Args:
            train_loader (torch.DataLoader): Loader with training data.
            learning_rate (float): Learning rate for training.
            n_epochs (int): Number of maximum epochs to train the model.
            train_patience (int): Early stopping patience for training loss.
            test_patience (int): Early stopping patience for test loss.
            test_loader (torch.DataLoader): If available, loader with test data.
        Return:
            epoch_hist (dict): Training history.
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
        Args:
            loader (torch.DataLoader): loader with test data
            device (torch.device): CUDA or CPU
        Return:
            test_dict (dict): dictionary with test metrics
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


