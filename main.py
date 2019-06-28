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

from single_directions_tools import mnist_mlp

model_and_output_directory = './models_and_output/'
training_data_directory = './training_data/'
device = 'cpu'

# Fraction of data that will be used for testing. Use less to save on
# time/computation.
data_fraction = .2
# If a unit has n ablatable units, units will be ablated in ablation_steps
# evenly spaced steps over the interval 1 to n. Use a smaller number to save
# on time/computation. Default step size is 1.
ablation_steps = 20
# Run the analysis
ablation_data = mnist_mlp.run_analyses(
    model_and_output_directory=model_and_output_directory,
    training_data_directory=training_data_directory,device=device,
    data_fraction=data_fraction, ablation_steps=ablation_steps)

# Ablation analysis on MNIST data
# ablate(mnist_generalization_models_path)
