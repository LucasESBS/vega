import torch
import numpy as np


class GelNet:
    """ 
    GelNet regularizer for linear decoder [Sokolov2016]_.
    If ``P`` is set to Identity matrix, this is Elastic net.
    ``d`` needs to be a `{0,1}`-matrix.
    If ``lamda1`` is 0, this is a L2 regularization. 
    If ``lambda2`` is 0, this is a L1 regularization.
    
    Needs to be sequentially used in training loop.

    Example
        >>> loss = MSE(X_hat, X)
        # Compute L2 term
        >>> loss += GelNet.quadratic_update(self.decoder.weight)
        >>> loss.backward()
        >>> optimizer.step()
        # L1 proximal operator update
        >>> GelNet.proximal_update(self.decoder.weight)
    
    Parameters
    ----------
    lambda1
        L1-regularization coefficient
    lambda2
        L2-regularization coefficient
    P
        Penalty matrix (eg. Gene network Laplacian)
    d
        Domain knowledge matrix (eg. mask)
    lr
        Learning rate
    """
    def __init__(self,
                 lambda1: float,
                 lambda2: float,
                 P: np.ndarray,
                 d: np.ndarray = None,
                 lr: float = 1e-3,
                 use_gpu: bool = False):
        self.l1 = lambda1
        self.l2 = lambda2
        if P is not None:
            self.P = torch.FloatTensor(P)
        if d is not None:
            d = torch.Tensor(d).bool()
        self.d = d
        self.lr = lr
        self.dev = torch.device('cuda') if use_gpu else torch.device('cpu')

    def quadratic_update(self, weights):
        """ 
        Computes the L2 term of GelNet 
        
        Parameters
        ----------
        weights
            Layer's weight matrix 
        """
        l = torch.tensor(0)
        if self.l2 == 0:
            return l
        else:
            # Sum over columns
            #for k in range(weights.size(1)):
                #l += (weights[:,k].t().matmul(self.P).matmul(weights[:,k]))
            # Use einsum
            l = torch.einsum('bi,ij,jb', weights.t(), self.P, weights)
        return self.l2*l
        
    def proximal_update(self, weights):
        """
        Proximal operator for the L1 term inducing sparsity.
        
        Parameters
        ----------
        weights
            Layer's weight matrix
        """
        if self.l1 == 0:
            return
        else:
            norm = self.l1 * self.lr
            w = weights.data
            w_update = w.clone()
            w_geq = w_update > norm
            w_leq = w_update < -1.0*norm
            w_sparse = ~w_geq&~w_leq
            if self.d is not None:
                w_update[(self.d&w_geq)] -= norm
                w_update[(self.d&w_leq)] += norm
                w_update[(self.d&w_sparse)] = 0.
            else:
                w_update[w_geq] -= norm
                w_update[w_leq] += norm
                w_update[w_sparse] = 0.
            weights.data = w_update
            return 

class LassoRegularizer:
    """ 
    Lasso (L1) regularizer for linear decoder.
    Similar to [Rybakov2020]_ lasso regularization.

    Parameters
    ----------
    lambda1
        L1-regularization coefficient
    d
        Domain knowledge matrix (eg. mask)
    lr
        Learning rate
    """
    def __init__(self, 
                lambda1: float,
                lr: float,
                d: np.ndarray = None,
                use_gpu: bool = False):
        self.l1 = lambda1
        self.lr = lr
        if d is not None:
            d = torch.Tensor(d).bool()
        self.d = d
        self.dev = torch.device('cuda') if use_gpu else torch.device('cpu')

    def quadratic_update(self, weights):
        """ Not applicable (identity) """
        return torch.tensor(0)
    
    def proximal_update(self, weights):
        """
        Proximal operator for the L1 term inducing sparsity.

        Parameters
        ----------
        weights
            Layer's weight matrix
        """
        if self.l1 == 0:
            return
        else:
            norm = self.l1 * self.lr
            w = weights.data
            w_update = w.clone()
            #norm_w = norm * torch.ones(w.size(), device=self.dev)
            #pos = torch.min(norm_w, norm * torch.clamp(w, min=0))
            #neg = torch.min(norm_w, -1.0 * norm * torch.clamp(w, max=0))
            #if self.d is not None:
                #w_update[self.d] = w[self.d] - pos[self.d] + neg[self.d]
            #else:
                #w_update = w - pos + neg
            w_geq = w_update > norm
            w_leq = w_update < -1.0*norm
            w_sparse = ~w_geq&~w_leq
            if self.d is not None:
                w_update[(self.d&w_geq)] -= norm
                w_update[(self.d&w_leq)] += norm
                w_update[(self.d&w_sparse)] = 0.
            else:
                w_update[w_geq] -= norm
                w_update[w_leq] += norm
                w_update[w_sparse] = 0.
            weights.data = w_update
            return 
    
         
                 
