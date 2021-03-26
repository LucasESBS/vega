#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom modules and layers for VEGA.

Acknowledgements:
Customized Linear from Uchida Takumi with modifications.
https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
Masked decoder based on LinearDecoderSCVI.
https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/modules/_base/__init__.py  
"""

import math
import numpy
from typing import Iterable, List
import torch
import torch.nn as nn
from scvi.nn import FCLayers, one_hot


class DecoderVEGA(nn.Module):
    """
    Masked linear decoder for VEGA.
    """
    def __init__(self, 
                mask,
                n_cat_list: Iterable[int] = None,
                n_continuous_cov: int = 0,
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                bias: bool = False
                ):
        super(DecoderVEGA, self).__init__()
        self.n_input = mask.shape[1]
        self.n_output = mask.shape[0]
        # Mean and dropout decoder - dropout is fully connected
        self.px_scale = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov = n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0)
        self.px_dropout = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov = n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """ Forward pass through VEGA's decoder """
        raw_px_scale = self.px_scale(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout
    
class SparseLayer(nn.Module):
    """
    Sparse Layer class. Inspired by SCVI 'FCLayers' but constrained to 1 layer.
    
    Parameters:
    -----------
    """
    def __init__(self,
                mask: numpy.ndarray,
                n_cat_list: Iterable[int] = None,
                n_continuous_cov: int = 0,
                use_activation: bool = False,
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                bias: bool = True,
                dropout_rate: float = 0.1,
                activation_fn: nn.Module = None
                ):
        # Initialize custom sparse layer
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.n_continuous_cov = n_continuous_cov
        self.cat_dim = sum(self.n_cat_list)
        mask_with_cov = numpy.vstack((mask, numpy.ones((self.n_continuous_cov+self.cat_dim, mask.shape[1]))))
        self.sparse_layer = nn.Sequential(
                                    CustomizedLinear(mask_with_cov),
                                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                                    if use_batch_norm else None,
                                    nn.LayerNorm(n_out, elementwise_affine=False)
                                    if use_layer_norm
                                    else None,
                                    activation_fn() if use_activation else None,
                                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
                                    )
    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on x for sparse layer.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layer in self.sparse_layer:
            if layer is not None:
                if isinstance(layer, nn.BatchNorm1d):
                    if x.dim() == 3:
                        x = torch.cat(
                            [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                        )
                    else:
                        x = layer(x)
                else:
                    if isinstance(layer, CustomizedLinear):
                        if x.dim() == 3:
                            one_hot_cat_list_layer = [
                                o.unsqueeze(0).expand(
                                    (x.size(0), o.size(0), o.size(1))
                                )
                                for o in one_hot_cat_list
                            ]
                        else:
                            one_hot_cat_list_layer = one_hot_cat_list
                        x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                    x = layer(x)
        return x 
    

class CustomizedLinearFunction(torch.autograd.Function):
    """
    Autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Args:
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


