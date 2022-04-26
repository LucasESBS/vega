import torch
from torch.nn import functional as F
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

class EarlyStopping:
    """
    Early stops the training if training/validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0,mode='train'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode = mode
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif self.mode=='valid' and score <= self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode=='train' and score <= self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

### -- LOSSES -- ###
class AnnealingVAELoss:
    def __init__(self, anneal_start, anneal_time, beta_start):
        """ Monotonic annealing of beta-VAE loss. From 0 to 1.
        
        If you want to use a standard beta-VAE loss, simply set beta_start !=0 and don't call
        update_beta()"""
        self.anneal_start = anneal_start
        self.anneal_time = anneal_time
        self.beta = beta_start

    def __call__(self, y_pred, y_true, mu, logvar):
        """ Compute loss."""
        # Compute VAE Loss as MSE() + beta*KL()
        rec = F.mse_loss(y_pred, y_true, reduction='sum')
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        loss = torch.mean(rec + self.beta * kld)
        return loss

    def update_beta(self, epoch, verbose=True):
        """ Update beta value"""
        # Update beta
        if epoch > self.anneal_start:
            self.beta = min((self.beta + 1./self.anneal_time), 1.)
        if verbose:
            print('New beta:', self.beta)
        return

### -- Weight Clipper --- ###
class WeightClipper(object):
    """ Clip weight of network every k epochs to constrain to positive weights """
    def __init__(self, frequency=5):
        """
        Args:
            frequency (int): Clipping weights after N epochs.
        """
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0)
            module.weight.data = w

## -- CrossValidation -- ##
class KFoldTorch:
    def __init__(self, cv=10, lr=1e-2, n_epochs=10, train_p=10, test_p=10, num_workers=0, save_all=True, save_best=False, path_dir='./', model_prefix='trained_model_'):
        """ 
        Implements startified kfold CV for training torch neural nets. This supposes your model has a 'train_model()' method that is callable.
        Args:
            cv (int): number of folds
            lr (float): learning rate for training models
            n_epochs (int): number of epoch for training
            train_p (int): patience for early stopper (train loss)
            test_p (int): patience for early stopper (test loss)
            num_workers (int): number of cpu to loader
            save_all (bool): If True, save model for each fold
            save_best (bool): If True, save best model over the Kfolds
            path_dir (str): directory where to save the model(s)
            model_prefix (str): prefix for naming the models when saved (give informative name for experiment)
        """
        self.cv = cv
        #self.return_model = return_model
        # Init result dict
        self.cv_res_dict = {c:{} for c in range(self.cv)}
        self.lr = lr
        self.n_epochs=n_epochs
        #self.best_metric=best_metric
        # If save_best, save_best model. If save_all, save model for all folds.
        self.save_best = save_best
        self.save_all = save_all
        self.train_p = train_p
        self.test_p = test_p
        self.num_work = num_workers
        # Directory where to save all models
        self.path_dir = path_dir if path_dir.endswith('/') else path_dir+'/'
        self.model_prefix = model_prefix
        print('Model(s) will be saved at %s using %s as prefix'%(self.path_dir, self.model_prefix), flush=True) 
    
    def train_kfold(self, blank_model, model_params, dataset, batch_size, drop_last_batch=True):
        """ 
        Run the Kfold cross validation.
        Args:
            blank_model (torch.nn.Module): Pytorch model class
            model_params (dict): Dictionary with parameters for initializing model
            dataset (torch.Dataset): Dataset containing data for training
            batch_size (int): Number of samples per batch
            drop_last_batch (bool): Whether to drop the last batch in case it has a different number of samples.
        """
        kfold = StratifiedKFold(n_splits=self.cv, shuffle=True)
        best_val_loss = 999999
        best_cv = 0
        best_model = None
        for i, (train_idx, test_idx) in enumerate(kfold.split(dataset, dataset.targets)):
            # Initialize model
            model = blank_model(**model_params).to(model_params['device'])
            print('Training fold %d'%(i), flush=True)
            train_ds = torch.utils.data.Subset(dataset, train_idx)
            test_ds = torch.utils.data.Subset(dataset, test_idx)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_batch, num_workers=self.num_work)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_batch, num_workers=self.num_work)
            # Train model
            epoch_hist = model._train_model(train_loader, test_loader=test_loader, n_epochs=self.n_epochs, learning_rate=self.lr, train_patience=self.train_p, test_patience=self.test_p, save_model=False)
            # Save attributes
            self.cv_res_dict[i]['history'] = epoch_hist
            if 'valid_loss' in epoch_hist.keys():
                self.cv_res_dict[i]['best_valid_loss'] = np.min(epoch_hist['valid_loss'])
            #if self.return_model:
                #self.cv_res_dict[i]['model'] = model
                if best_val_loss > np.min(epoch_hist['valid_loss']):
                    best_val_loss = np.min(epoch_hist['valid_loss'])
                    best_cv = i
                    best_model = model
            # Save if save all
            if self.save_all:
                full_path = self.path_dir+self.model_prefix+'fold_'+str(i)+'.pt'
                print('Saving model at %s'%(full_path), flush=True)
                torch.save(model.state_dict(), full_path)

            # delete model from memory ?
            del model
            if model_params['device'] == torch.device('cuda'):
                torch.cuda.empty_cache()

        self.best_cv = best_cv
        print('Best Fold: %d'%(self.best_cv), flush=True)
        # Save best model to path
        if self.save_best and not self.save_all:
            path_best = self.path_dir+self.model_prefix+'best_fold.pt'
            print('Saving best model at %s'%(path_best), flush=True)
            torch.save(best_model.state_dict(), path_best)
        elif self.save_best and self.save_all:
            print('Best model already saved for fold %d'%(self.best_cv), flush=True)

        return

# --- General purpose datasets --- #
class ClassificationDataset(torch.utils.data.Dataset):
    "Characterizes a classification dataset for PyTorch"
    def __init__(self, data, targets):
        "Initialization"
        self.targets = targets
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.targets)

    def __getitem__(self, index):
        "Generates samples of data"
        # Load data and get label
        X = self.data[index]
        y = self.targets[index]

        return X, y.long()

class UnsupervisedDataset(torch.utils.data.Dataset):
    "Characterizes a unsupervised learning dataset for PyTorch"
    def __init__(self, data, targets=None):
        "Initialization"
        self.targets = targets
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        "Generates samples of data"
        # Load data
        X = self.data[index]
        return X
