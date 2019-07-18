# An exercise to familiarize myself with applied deep learning using PyTorch
# by replicating Morcos et al.'s 2018 ICLR paper On the Importance of
# Single Directions for Generalization. Assumes access to trained models.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import single_directions_tools as tools
import model_definitions as models

model_and_output_directory = './models_and_output/'
training_data_directory = './training_data/'
device = 'cpu'

# Fraction of data that will be used for testing. Use less to save on
# time/computation.
data_fraction = .2
# If a unit has n ablatable units, units will be ablated in ablation_steps
# evenly spaced steps over the interval 1 to n. Use a smaller number to save
# on time/computation. Default step size is 1.
ablation_steps = 15
# Number of replicates
n_repetitions = 5

# MLP Analyses
mlp_data_path = model_and_output_directory + 'mnist_mlp/'

# Download and load MNIST data
mnist_train_data = datasets.MNIST(
    training_data_directory,
    train=True,
    download=True,
    transform=transforms.ToTensor())
mnist_train_loader = torch.utils.data.DataLoader(
    dataset=mnist_train_data, shuffle=True)
mnist_test_data = datasets.MNIST(
    training_data_directory,
    train=False,
    download=True,
    transform=transforms.ToTensor())
mnist_test_loader = torch.utils.data.DataLoader(
    dataset=mnist_test_data, shuffle=False)

# MLP Parameters
input_size = 28 * 28
output_size = 10
n_units_per_layer = {
    'selectivity': 128,
    'generalization': 512,
    'early_stopping': 2048,
    'dropout': 2048
}
# Loss function
criterion = nn.CrossEntropyLoss()

mlp = models.MLP(input_size, n_units_per_layer['generalization'], output_size)
ablation_layers = [
    ['linears.0.weight', 'linears.0.bias'],
    ['linears.1.weight', 'linears.1.bias']
]

mlp_generalization_path = mlp_data_path \
                          + 'generalization/'
mlp_generalization_data_path = mlp_generalization_path + 'ablation_data/'
mlp_generalization_data_pickle_root = 'mnist_mlp_generalization_'

# Ablation analysis
mlp_generalization_ablation_data_path = mlp_generalization_data_path \
                                        + mlp_generalization_data_pickle_root \
                                        + 'ablation.pkl'
# If data exist, load them. Otherwise run the analyses
if os.path.exists(mlp_generalization_ablation_data_path):
    mlp_ablation_data = pd.read_pickle(mlp_generalization_ablation_data_path)
else:
    # Run ablation analysis
    mlp_ablation_data = tools.ablation_test(
        mlp, mnist_train_loader, criterion,
        mlp_generalization_path, device=device,
        data_fraction=data_fraction, ablation_steps=ablation_steps,
        n_repetitions=n_repetitions)
    # Save data
    mlp_ablation_data.to_pickle(mlp_generalization_ablation_data_path)

# Noise injection analysis
mlp_generalization_noise_data_path = mlp_generalization_data_path \
                                     + mlp_generalization_data_pickle_root \
                                     + 'noise.pkl'
# Logarithmic range over which noise should be scaled (10^x : 10^y)
noise_range = [-1.5, 1]
# If data exist, load them. Otherwise run the analyses
if os.path.exists(mlp_generalization_noise_data_path):
    mlp_noise_data = pd.read_pickle(mlp_generalization_noise_data_path)
else:
    # Run ablation analysis
    mlp_noise_data = tools.ablation_test(
        mlp, mnist_train_loader, criterion,
        mlp_generalization_path, device=device,
        data_fraction=data_fraction, ablation_steps=ablation_steps,
        n_repetitions=n_repetitions, ablation_type='noise',
        noise_scale=noise_range)
    # Save data
    mlp_noise_data.to_pickle(mlp_generalization_noise_data_path)

# CNN Analyses
# Download and transform training data
# Convert to tensor and normalize to be in range 0-1. Compose allows chaining (
# composition) of multiple transformations.
cifar_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_train_data = datasets.CIFAR10(
    training_data_directory,
    train=True,
    download=True,
    transform=cifar_transform)
cifar_train_loader = torch.utils.data.DataLoader(
    dataset=cifar_train_data,
    shuffle=True)
cifar_test_data = datasets.CIFAR10(
    training_data_directory,
    train=False,
    download=True,
    transform=cifar_transform)
cifar_test_loader = torch.utils.data.DataLoader(
    dataset=cifar_test_data, shuffle=False)

# CNN Parameters
n_conv_layers = 11
input_channels = 3
input_size = 32
batch_norm_flag = True
# Size of fully-connected layer
fc_layer_size = 256
# Dictionary containing layers parameters
conv_layers = {
    'channels': [64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
    'stride': [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
    'kernel': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'pad': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
# Calculate size of output
output_size = input_size
for layer_i in range(n_conv_layers):
    output_size, _ = tools.conv_output_shape((output_size, output_size),
                                     conv_layers['kernel'][layer_i],
                                     conv_layers['stride'][layer_i],
                                     conv_layers['pad'][layer_i])
output_size *= output_size
output_size *= conv_layers['channels'][n_conv_layers-1]

cnn = models.CifarNN(conv_layers, output_size, fc_layer_size)
tools.test(cnn, cifar_train_loader, criterion, [], [])
