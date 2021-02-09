#!/usr/bin/env python3

# Module for VEGA

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from .customized_linear import CustomizedLinear
from .utils import *
from .learning_utils import *
import scanpy as sc
from scipy import sparse

class VEGA(torch.nn.Module):
    def __init__(self, pathway_mask, positive_decoder=False, **kwargs):
        """ 
        Constructor for class VEGA (VAE Enhanced by Gene Annotations).

        VEGA is a VAE-based model for inference of Gene Module Variables (GMVs) activities. It uses
        a binary matrix initialized from a .gmt file to mask connection in a linear decoder in order
        to provide interpretability over each latent variable of the model, providing an easy way to 
        project data into the interpretable latent space and perform differential testing.

        Args:
            pathway_mask (np.array): Gene module mask.
            positive_decoder (bool): Whether to constrain decoder to positive weights.
            **kwargs: Arbitrary keyword arguments.

        """
        super(VEGA, self).__init__()
        self.pathway_mask = pathway_mask
        self.n_pathways = self.pathway_mask.shape[1]
        self.n_genes = self.pathway_mask.shape[0]
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.beta = kwargs.get("beta", 0.01)
        self.save_path = kwargs.get('path_model', "trained_vae.pt")
        self.dropout = kwargs.get('dropout', 0.2)
        self.pos_dec = positive_decoder
        self.encoder = nn.Sequential(nn.Linear(self.n_genes, 800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(800,800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout))
        self.mean = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.decoder = CustomizedLinear(self.pathway_mask.T)
        # Constraining decoder or not
        if self.pos_dec:
            print('Constraining decoder to positive weights', flush=True)
            self.decoder.reset_params_pos()
            self.decoder.weight.data *= self.decoder.mask        


    def encode(self, X):
        """ 
        Encode data in latent space.

        Encoder function for VEGA model.

        Args:
            X (torch.tensor): input data

        Returns:
            (tuple): tuple containing:

                z (torch.tensor): Data in latent space\n
                mu (torch.tensor): Mean of variational posterior\n
                logvar (torch.tensor): Log-variance of variational posterior
        """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """ 
        Decode data from latent space.

        Decoder function for VEGA model.

        Args:
            z (torch.tensor): Data embedded in latent space.
        Returns:
            torch.tensor: Decoded data.
        """
        X_rec = self.decoder(z)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ 
        Sample latent space with reparametrization trick. First convert to standard deviation, then sample from standard normal dsitribution and get latent vector.

        Args:
            mu (torch.tensor): Mean of variational posterior.
            logvar (torch.tensor): Log-variance of variational posterior.
        Returns:
            torch.tensor: Sampled latent space.
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        """
        Same as encode, but only returns $z$ (no mu and logvar).

        Args:
            X (torch.tensor): Input data.
        Returns:
            torch.tensor: Embedded data in latent space.
        """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z
 
    def _average_latent(self, X):
        """
        Computes the average data vector in the latent space.

        Args:
            X (torch.tensor): Input data.
        Returns:
            torch.tensor: Mean latent vector.
        """
        z = self.to_latent(X)
        mean_z = z.mean(0)
        return mean_z

    def bayesian_diff_exp(self, adata1, adata2, n_samples=2000, use_permutations=True, n_permutations=1000, random_seed=False):
        """ 
        Run Bayesian differential testingin latent space.

        Method based on scVI Bayesian differential expression procedure [Lopez, 2018]_. Compares activity of 
        each latent variable in the latent space and computes Bayes factor for each of them.

        Args:
            adata1 (Anndata): Single-cell dataset of group 1.
            adata2 (Anndata): Single-cell dataset of group 2.
            n_samples (int): Number of samples to draw from the latent space.
            use_permutations (bool): Whether to use permutations when computing the double integral.
            n_permutations (int): Number of permutations for MC integral.
            random_seed (int): Seed for reproducibility.
        Returns:
            dict: Dictionary with results.
        """
        self.eval()
        # Set seed for reproducibility
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        epsilon = 1e-12
        # Sample cell from each condition
        idx1 = np.random.choice(np.arange(len(adata1)), n_samples)
        idx2 = np.random.choice(np.arange(len(adata2)), n_samples)
        # To latent
        z1 = self.to_latent(torch.Tensor(adata1[idx1,:].X)).detach().numpy()
        z2 = self.to_latent(torch.Tensor(adata2[idx2,:].X)).detach().numpy()
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

    def _scale_sampling(self, arr1, arr2, n_perm=1000):
        """
        Use permutation to better estimate double integral (create more pair comparisons)
        Inspired by scVI (Lopez et al., 2018)

        Args:
            arr1 (np.array): array with data of group 1.
            arr2 (np.array): array with data of group 2.
            n_perm (int): number of permutations.
        Returns:
            np.array: Samples for group 1.
            np.array: Samples for group 2.
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
        Returns:
            tuple: Tuple containing:
                X_rec (torch.tensor): reconstructed data \n
                mu (torch.tensor): mean of variational posterior\n
                logvar (torch.tensor): log-variance of variational posterior\n
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
            torch.tensor: Loss value for current batch.
        """
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta*kld)

    def train_model(self, train_loader, learning_rate, n_epochs, train_patience, test_patience, test_loader=False, save_model=True):
        """
        Method to train VEGA.

        Args:
            train_loader (torch.DataLoader): Loader with training data.
            learning_rate (float): Learning rate for SGD.
            n_epochs (int): Number of maximum epochs to train the model.
            train_patience (int): Early stopping patience for training loss.
            test_patience (int): Early stopping patience for test loss.
            test_loader (torch.DataLoader): If available, loader with test data.
            save_model (bool): If True, save model after training to path attribute of VEGA.
        Returns:
            dict: Training history
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
                if self.pos_dec:
                    self.decoder.apply(clipper)
            # Get epoch loss
            epoch_loss = loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_hist['train_loss'].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict = self.test_model(test_loader)
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
        # Save model
        if save_model:
            print('Saving model to ...', self.save_path)
            torch.save(self.state_dict(), self.save_path)

        return epoch_hist

    def test_model(self, loader):
        """
        Test model on input loader.

        Args:
            loader (torch.DataLoader): Loader with test data.
        Returns:
            dict: Dictionary with test metrics.
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
        test_dict['loss'] = loss/(len(loader)*loader.batch_size)
        return test_dict


