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
from vega.layers import CustomizedLinear
from vega.utils import *
from vega.learning_utils import *
import scanpy as sc
from scipy import sparse

class VEGA(torch.nn.Module):
    def __init__(self, adata,
                gmt_paths=None, 
                add_nodes=1,
                min_genes=0,
                max_genes=5000,
                positive_decoder=True, 
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
        kwargs:
            dev (torch.device): using CPU or CUDA.
            beta (float): weight for KL-divergence.
            dropout (float): dropout rate in model.
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
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.beta_ = kwargs.get("beta", 0.0001)
        self.dropout_ = kwargs.get('dropout', 0.3)
        self.pos_dec_ = positive_decoder
        self.epoch_history = {}
        # Model architecture
        self.encoder = nn.Sequential(nn.Linear(self.n_genes, 800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_),
                                    nn.Linear(800,800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_))
        self.mean = nn.Sequential(nn.Linear(800, self.n_gmvs), 
                                    nn.Dropout(self.dropout_))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_gmvs), 
                                    nn.Dropout(self.dropout_))
        self.decoder = CustomizedLinear(self.gmv_mask.T)
        self.is_trained_ = False
        # Constraining decoder to positive weights or not
        if self.pos_dec_:
            print('Constraining decoder to positive weights', flush=True)
            self.decoder.reset_params_pos()
            self.decoder.weight.data *= self.decoder.mask        

    def __repr__(self):
        print("VEGA model with the following paramaters: \nn_GMVs: {}, dropout_rate:{}, beta:{}, positive_decoder:{}".format(self.n_gmvs, self.dropout_, self.beta_, self.pos_dec_))
        print("Model is trained: {}".format(self.is_trained_))
        return

    def save(self, path, save_adata=False, save_history=False):
        """ 
        Save model parameters to input directory. Saving Anndata object and training history is optional.
        Args:
            path (str): Path to save directory.
            save_adata (bool): Whether to save the Anndata object in the save directory.
            save_history (bool): Whether to save the training history in the save directory.
        """
        attr = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attr = [a for a in attr if not (a[0].startswith("__") and a[0].endswith("__")]
        attr_dict = {a[0]:a[1] for a in attr if a[0][-1]=='_'}
        # Save
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
        model = cls(adata, **attr)
        # Reload Anndata
        if not adata:
            try:
                model.adata = sc.read(os.path.join(path, 'anndata.h5ad'))
            except:
                FileNotFoundError('No Anndata object was passed or found in input directory.')
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

    def encode(self, X):
        """ 
        Encode data in latent space.
        Args:
            X (torch.tensor): input data
        Return:
            z (torch.tensor): data in latent space
            mu (torch.tensor): mean of variational posterior
            logvar (torch.tensor): log-variance of variational posterior
        """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """ 
        Decode data from latent space.
        Args:
            z (torch.tensor): data embedded in latent space
        Return:
            X_rec (torch.tensor): decoded data
        """
        X_rec = self.decoder(z)
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
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        """
        Same as encode, but only returns z (no mu and logvar).
        Args:
            X (torch.tensor): input data
        Return:
            z (torch.tensor): embedded data in latent space
        """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z
 
    def _average_latent(self, X):
        """
        Computes the average data vector in the latent space.
        Args:
            X (torch.tensor): input data
        Return:
            mean_z (torch.tensor): mean latent vector
        """
        z = self.to_latent(X)
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
            group1 = self.adata.obs[groupby].unique()
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
        self.eval()
        # Set seed for reproducibility
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        epsilon = 1e-12
        # Subset data
        adata1, adata2 = adata.X[cell_idx1,:], adata.X[cell_idx2,:]
        # Sample cell from each condition
        idx1 = np.random.choice(np.arange(len(adata1)), n_samples)
        idx2 = np.random.choice(np.arange(len(adata2)), n_samples)
        # To latent
        z1 = self.to_latent(torch.Tensor(adata1[idx1,:])).detach().numpy()
        z2 = self.to_latent(torch.Tensor(adata2[idx2,:])).detach().numpy()
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
    def _scale_sampling(self, arr1, arr2, n_perm=1000):
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

    def forward(self, X):
        """
        Forward pass through full network.
        Args:
            X (torch.tensor): input data
        Return:
            X_rec (torch.Tensor): reconstructed data
            mu (torch.tensor): mean of variational posterior
            logvar (torch.tensor): log-variance of variational posterior
        """
        z, mu, logvar = self.encode(X)
        X_rec = self.decode(z)
        return X_rec, mu, logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        """ 
        Custom loss for VAE.
        Args:
            y_pred (torch.tensor): Reconstructed data.
            y_true (torch.tensor): Real data.
            mu (torch.tensor): Mean of variational posterior.
            logvar (torch.tensor): Log-variance of variational posterior.
        Return:
            loss value for current batch
        """
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta_*kld)

    def train(self, learning_rate=1e-4, n_epochs=500, train_size=1., batch_size=128, shuffle=True, **kwargs):
        """ 
        Main method to train VEGA.
        Args:
            learning_rate (float): Learning rate.
            n_epochs (int): Number of epochs to train model.
            train_size (float): A number between 0 and 1 to indicate the proportion of training data. Test size is set to 1-train_size.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle samples or not.
            **kwargs: Other keyword arguments of the _train_model() method, like the early stopping patience.
        """
        train_patience=kwargs.get('train_patience', 10)
        test_patience=kwargs.get('test_patience', 10)
        # Create dataloader from Anndata
        train_data, test_data = _anndata_splitter(self.adata, train_size=train_size)
        train_loader = _anndata_loader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = _anndata_loader(test_data, batch_size=batch_size, shuffle=shuffle) if test_data else False
        # Call training method
        self.epoch_history = self._train_model(train_loader=train_loader,
                                            learning_rate=learning_rate,
                                            n_epochs=n_epochs,
                                            train_patience=train_patience,
                                            test_patience=test_patience,
                                            test_loader=test_loader
                                            )
        return

    def _train_model(self, train_loader, learning_rate, n_epochs, train_patience=10, test_patience=10, test_loader=False):
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
        clipper = WeightClipper(frequency=1)
        # Train
        for epoch in range(n_epochs):
            loss_value = 0
            self.train()
            for x_train in train_loader:
                x_train = x_train.to(self.dev)
                optimizer.zero_grad()
                x_rec, mu, logvar = self.forward(x_train)
                loss = self.vae_loss(x_rec, x_train, mu, logvar)
                loss_value += loss.item()
                loss.backward()
                optimizer.step()
                if self.pos_dec_:
                    self.decoder.apply(clipper)
            # Get epoch loss
            epoch_loss = loss_value / len(train_loader.dataset)
            epoch_hist['train_loss'].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict = self._test_model(test_loader)
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
    
    def _test_model(self, loader):
        """
        Test model on input loader.
        Args:
            loader (torch.DataLoader): loader with test data
        Return:
            test_dict (dict): dictionary with test metrics
        """
        test_dict = {}
        loss = 0
        loss_func = self.vae_loss
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.dev)
                reconstruct_X, mu, logvar = self.forward(data)
                loss += loss_func(reconstruct_X, data, mu, logvar).item()
        test_dict['loss'] = loss/len(loader.dataset)
        return test_dict


