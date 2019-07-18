import os
import itertools
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class MLP(nn.Module):
    """Two-layer MLP

    Class for creating two-layer MLP for the MNIST. Output is softmax across
    10 classes. It is assumed that the network will only be ever used in eval()
    mode, and thus dropout layers will be ignored.

    Parameters
    ----------

    input_size : int
        Size of input
    n_units_per_layer : int
        Number of units per layer
    output_size : int
        Number of outputs

    """

    def __init__(self, input_size, n_units_per_layer, output_size):
        super(MLP, self).__init__()
        # Set layer parameters
        self.input_size = input_size
        # Linear transformation: out = input x A' + b
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, n_units_per_layer)])
        self.linears.append(nn.Linear(n_units_per_layer, n_units_per_layer))
        self.linears.append(nn.Linear(n_units_per_layer, output_size))
        # List of units stored as pairs of integers
        self.unit_list = list(itertools.product(
            [0, 1], list(range(n_units_per_layer))))
        # Initialize list of units to ablate
        self.ablate(init=True)
        # Stores activations
        self.activations = [[], []]
        # Flag to store activations
        self.store_activations = False
        # Holds standard deviations of activations across entire training
        # set for each unit. Used when injecting noise. Each entry is an n
        # x 1 list of variances, in which n = number of units.
        self.sd = []
        # Scales noise injection
        self.sd_scale = 0

        # Define forward pass
    def forward(self, x):
        # Reshape input image into 1 x n row vector
        x = x.view(-1, self.input_size)
        # First layer
        # Rectified linear unit (ReLU) activation
        x = F.relu(self.linears[0](x))
        # If injecting noise
        if self.sd_scale:
            x = x + torch.from_numpy(np.random.normal(0, self.sd[0])).float() \
                * self.sd_scale
        # If saving activations
        if self.store_activations:
            self.activations[0].append(x.squeeze().tolist())
        # Second layer
        x = F.relu(self.linears[1](x))
        if self.sd_scale:
            x = x + torch.from_numpy(np.random.normal(0, self.sd[1])).float() \
                * self.sd_scale
        if self.store_activations:
            self.activations[1].append(x.squeeze().tolist())
        # Softmax on output layer
        return F.log_softmax(self.linears[2](x), dim=1)

    def ablate(self, init=False, layer=None, unit=None):
        """Method to ablate a unit.

        Units are ablated in random order from a specified set of layers.
        Ablation is achieved by setting a unit's input weights and bias to
        zero. If values aren't passed, will pop value from
        self.ablation_list and return true. If ablation_list is empty,
        will return false and re-initialize ablation_list.

        Parameters
        ----------
        init : boolean
            If True, will initialize self.ablation_list with a random
            ordering of candidate units.
        layer : int
            layer number. If not provided, will pop first value pair in
            self.ablation_list.
        unit : int
            unit number. If not provided, will pop first value pair in
            self.ablation_list.

        Returns
        -------
        boolean
            False if ablation_list is empty, True if ablation_list is not empty
        """
        if init:
            self.ablation_list = list(np.random.permutation(self.unit_list))
        else:
            # If layer or unit aren't provided, use ablation_list
            if layer is None or unit is None:
                # If ablation_list isn't empty, pop value and return True
                if self.ablation_list:
                    unit = self.ablation_list.pop()
                    layer, unit = unit[0], unit[1]
                    self.linears[layer].weight[unit] = 0
                    self.linears[layer].bias[unit] = 0
                    return True
                # If ablation_list is empty, re-initialize it and return False
                else:
                    self.ablate(init=True)
                    return False
            else:
                self.linears[layer].weight[unit] = 0
                self.linears[layer].bias[unit] = 0


# Define network
class CifarNN(nn.Module):
    """Convolutional net

    Class for creating deep CNN with fully-connected final layer. Output is
    softmax across10 classes. It is assumed that the network will only be
    ever used in eval() mode

    Parameters
    ----------

    conv_layers : dict
        Dictionary with keys 'channels', 'stride', 'kernel', and 'pad',
        each of which's value is a list of length n layers, each of which
        describes the respective parameter for each convolutional layer.
    conv_output_size : int
        Size of output from final convolutional layer
    fc_layer_size : int
        Number of units in fully connected layer
    n_input_channels : int
        Number of input channels. Default = 3.
    batch_norm_flag : boolean
        Flag determining whether to include batch norm. Default = True.

    """
    def __init__(self, conv_layers, conv_output_size, fc_layer_size,
                 n_input_channels=3, batch_norm_flag=True):
        super(CifarNN, self).__init__()
        self.n_conv_layers = len(conv_layers['channels'])
        self.conv_output_size = conv_output_size
        self.batch_norm_flag = batch_norm_flag
        # Initialize layers
        for layer_i in range(self.n_conv_layers):
          if layer_i == 0:
            self.conv = nn.ModuleList(
                [nn.Conv2d(
                    n_input_channels,
                    conv_layers['channels'][layer_i],
                    conv_layers['kernel'][layer_i],
                    conv_layers['stride'][layer_i],
                    conv_layers['pad'][layer_i])
                ])
            if batch_norm_flag:
              self.bn = nn.ModuleList(
                  [nn.BatchNorm2d(conv_layers['channels'][layer_i])]
                  )
          else:
            self.conv.append(
                nn.Conv2d(
                    conv_layers['channels'][layer_i-1],
                    conv_layers['channels'][layer_i],
                    conv_layers['kernel'][layer_i],
                    conv_layers['stride'][layer_i],
                    conv_layers['pad'][layer_i])
                )
            if batch_norm_flag:
              self.bn.append(
                  nn.BatchNorm2d(conv_layers['channels'][layer_i])
                  )
        self.fc1 = nn.Linear(self.conv_output_size, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, 10)

        # # List of units that can be ablated. Default is final three layers.
        # self.unit_list =
        # # Initialize list of units to ablate
        # self.ablate(init=True)
        # # Stores activations
        # self.activations = [[], []]
        # # Flag to store activations
        # self.store_activations = False
        # # Holds standard deviations of activations across entire training
        # # set for each unit. Used when injecting noise. Each entry is an n
        # # x 1 list of variances, in which n = number of units.
        # self.sd = []
        # # Scales noise injection
        # self.sd_scale = 0

    def forward(self, x):
        print(x)
        for layer_i in range(self.n_conv_layers):
            x = self.conv[layer_i](x)
            print('layer: ' + str(layer_i))
            print(x)
            if self.batch_norm_flag:
                x = self.bn[layer_i](x)
            x = F.relu(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
