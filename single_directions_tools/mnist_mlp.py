import os
import itertools
import time

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
        # Define forward pass

    def forward(self, x):
        # Reshape input image into 1 x n row vector
        x = x.view(-1, self.input_size)
        # Rectified linear unit (ReLU) activation
        x = F.relu(self.linears[0](x))
        x = F.relu(self.linears[1](x))
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


def test(model, data_loader, criterion, loss_vector, accuracy_vector,
         data_fraction=1, device='cpu'):
    """Test model on data

    Parameters
    ----------

    model : Net object
        Model to be tested
    data_loader : DataLoader object
        Data to test on
    criterion : loss function
        from torch.nn
    loss_vector : list
        Loss for dataset
    accuracy_vector : list
        Accuracy for dataset
    data_fraction : int
        Fraction of data to test on.
    device : str
        Device that models will be deployed to (cpu, gpu, etc)
    """
    n_trials_to_test = round(data_fraction * len(data_loader))
    # Set model to evaluation mode
    model.eval()
    # No need for gradient computations
    torch.no_grad()
    # Running collection of loss and outcome for each data sample
    val_loss, correct = 0, 0
    start_time = time.time()
    for i, (data, target) in enumerate(data_loader, 1):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        if not i % 1000:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(str(i) + ' out of ' + str(n_trials_to_test) + ' examples: '
                  + str(elapsed_time) + ' seconds')
        if i >= n_trials_to_test:
            break

    val_loss /= n_trials_to_test
    loss_vector.append(val_loss)

    accuracy = correct.to(torch.float32) / n_trials_to_test
    accuracy_vector.append(accuracy)


def ablation_test(model, data_loader, criterion, params_path,
                  ablation_data={}, device='cpu',
                  n_repetitions=10, data_fraction=1, ablation_steps=None):
    """Tests model as units are randomly ablated (zero'd)

    Units are ablated in random order from a specified set of layers.
    Ablation is achieved by setting a unit's input weights and bias to zero.

    Parameters
    ----------
    model : Net object
        Model to be ablated and tested
    data_loader : DataLoader object
        Data to test on
    criterion : loss function
        from torch.nn
    params_path : str
        Path to model or directory of model parameters and shuffled trial
        labels. ablate() is applied to all models in the directory. Shuffled
        trial labels are assumed to be in the same directory as the model(s).
    ablation_data : dictionary
        A dictionary in which keys = variables (e.g. accuracy, shuffle
        condition) and values are lists of data point values for that
        variable
    device : str
        Device that models will be deployed to (cpu, gpu, etc)
    n_repetitions : int
        Number of repetitions of ablation to run. Default = 10.
    data_fraction : float, (0, 1]
        Fraction of data that will be used for testing. Use less to save on
        time/computation. Default = 1.
    ablation_steps : int
        If a unit has n ablatable units, units will be ablated in
        ablation_steps evenly spaced steps over the interval 1 to n. Use a
        smaller number to save on time/computation. Default step size is 1.

    Returns
    -------
    ablation_data
    """
    # Keys/variables for ablation_data
    shuff_key = 'fraction of labels corrupted'
    loss_key = 'loss'
    accuracy_key = 'accuracy'
    n_units_ablated_key = 'units ablated'
    dropout_key = 'dropout fraction'
    rep_key = 'repetition'
    # Initialize ablation_data keys if it's empty
    if not ablation_data:
        ablation_data_keys = [shuff_key, loss_key, accuracy_key,
                              n_units_ablated_key, dropout_key, rep_key]
        for key in ablation_data_keys:
            ablation_data[key] = []
    # Check if model_path is a directory. If so, loop through each model
    # file in the directory and call ablate() recursively.
    if os.path.isdir(params_path):
        # List of models in directory
        files = [f for f in os.listdir(params_path) \
                 if os.path.isfile(os.path.join(params_path, f)) \
                 and 'model' in f]
        for model_name in files:
            print('Analyzing file: ' + model_name)
            ablation_test(model, data_loader, criterion, params_path +
                          model_name, ablation_data=ablation_data,
                          device=device, data_fraction=data_fraction,
                          n_repetitions=n_repetitions,
                          ablation_steps=ablation_steps)
        ablation_data
    else:
        # Path of directory containing data
        params_directory_path = params_path[0: params_path.rfind('/') + 1]
        # Filename of current model parameters
        params_filename = params_path[params_path.rfind('/') + 1:]
        # Shuffled labels. Find file with matching model parameter file
        # name substring
        shuffled_targets_filename = \
            params_filename[0: params_filename.find('epochs')] \
            + 'targets.pt'
        # Load shuffled labels
        data_loader.dataset.targets = torch.load(
            params_directory_path + shuffled_targets_filename)
        n_units = len(model.ablation_list)
        if ablation_steps is None:
            ablation_steps = n_units
        # Ablation counts on which model should be tested
        ablation_test_counts = np.linspace(0, n_units, ablation_steps).round()
        # Create values for ablation_curves
        shuffle_fraction = \
            float(params_filename[params_filename.find('shuff') + 5 :
                                  params_filename.find('shuff') + 8])
        dropout_fraction = \
            float(params_filename[params_filename.find('dropout') + 7 :
                                  params_filename.find('dropout') + 10])
        # Repeat ablation process for repetitions
        for rep_n in range(n_repetitions):
            # Load Model
            model.load_state_dict(torch.load(params_path, map_location=device))
            # Flag to continue ablating
            ablation_flag = True
            # Count of units ablated
            n_units_ablated = 0
            print('Repetition ' + str(rep_n + 1) + ' out of ' + str(
                n_repetitions))
            # Test, ablate, repeat
            while ablation_flag:
                if n_units_ablated in ablation_test_counts:
                    loss, accuracy = [], []
                    print('Ablated ' + str(n_units_ablated) + ' units of ' +
                          str(n_units))
                    # Test
                    test(model, data_loader, criterion, loss,
                         accuracy, data_fraction=data_fraction,
                         device=device)
                    # Add values to ablation_data
                    ablation_data[shuff_key].append(
                        shuffle_fraction)
                    ablation_data[loss_key].append(loss[0])
                    ablation_data[accuracy_key].append(accuracy[0].item())
                    ablation_data[n_units_ablated_key].append(n_units_ablated)
                    ablation_data[dropout_key].append(dropout_fraction)
                    ablation_data[rep_key].append(rep_n)
                # Ablate and set flag
                ablation_flag = model.ablate()
                # Increment ablation counter
                n_units_ablated += 1
    return ablation_data

def run_analyses(
        model_directory='./trained_models/',
        training_data_directory='./data',
        output_data_directory = './output_data/',
        device='cpu',
        data_fraction=1,
        ablation_steps=None):
    """Function for running analyses on MNIST MLPs

    Parameters
    ----------
    model_directory : str
        Path to base directory that contains models
    training_data_directory : str
        Path to base directory that contains data sets (e.g. MNIST)
    output_data_directory : str
        Path to directory that where analysis data sets will be saved
    device : str
        Device that models will be deployed to (cpu, gpu, etc)
    data_fraction : float, (0, 1]
        Fraction of data that will be used for testing. Use less to save on
        time/computation. Default = 1.
    ablation_steps : int
        If a unit has n ablatable units, units will be ablated in
        ablation_steps evenly spaced steps over the interval 1 to n. Use a
        smaller number to save on time/computation. Default step size is 1.

    Returns
    -------
    ablation_data : pandas DataFrame
        Contains results from ablation analysis
    """

    mlp_models_and_labels_path = model_directory + 'mnist_mlp/'

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
        dataset=mnist_test_data, shuffle=True)

    # Generalization analysis
    mlp = MLP(input_size, n_units_per_layer['generalization'], output_size)
    mlp_generalization_model_params_path = mlp_models_and_labels_path \
                                           + 'generalization/'
    ablation_layers = [
        ['linears.0.weight', 'linears.0.bias'],
        ['linears.1.weight', 'linears.1.bias']
    ]

    ablation_data = ablation_test(
        mlp, mnist_train_loader, criterion,
        mlp_generalization_model_params_path, device=device,
        data_fraction=data_fraction, ablation_steps=ablation_steps,
        n_repetitions=10)
    ablation_data = pd.DataFrame(ablation_data)
    data_save_path = output_data_directory + 'generalization_MNIST.pkl'
    ablation_data.to_pickle(data_save_path)

    sns.set()
    plt.figure()
    sns.lineplot(x='units ablated', y='accuracy',
                 hue='fraction of labels corrupted', data=ablation_data,
                 palette=sns.color_palette('Blues_d'))
