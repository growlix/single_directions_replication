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
                  n_repetitions=10, data_fraction=1, ablation_type='zero',
                  ablation_steps=None, noise_scale=[-1.5, 1]):
    """Tests model as units are randomly ablated (zeroed out or noise added)

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
        Path to parent directory for this analysis. Models and targets will
        be in params_path/models_and_targets; unit activations and standard
        deviations will be saved to params_path/activations; analysis
        results will be saved to params_path/ablation_data.
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
    ablation_type : string ('zero', 'noise')
        The type of ablation to perform. 'zero' (default) will zero out
        units. 'noise' will add noise, scaled by the empirical variance of
        the units' activation across the training set.
    ablation_steps : int
        If ablation_type is 'zero': If a unit has n ablatable units,
        units will be ablated in ablation_steps evenly spaced steps over
        the interval 1 to n. Use a smaller number to save on
        time/computation. Default step size is 1.
        If ablation_type is 'noise': Noise with zero mean and progressively
        increasing variance (scaled by the empirical variance of each
        units' activations will be added in logarithmically-spaced increments.
    noise_scale : list
        interval over which noise will be scaled logarithmically. over
        ablation_steps number of steps E.g. noise_scale = [-1.5, 1] and
        ablation_steps = 10 will yield 10 logarithmically spaced steps from
        10^-1.5 to 10^1. Default = [-1.5, 1].
    Returns
    -------
    ablation_data
    """
    # Paths
    path_models_and_targets = params_path + 'models_and_targets/'
    path_activations = params_path + 'activations/'
    # Keys/variables for ablation_data
    shuff_key = 'fraction of labels corrupted'
    loss_key = 'loss'
    accuracy_key = 'accuracy'
    n_units_ablated_key = 'units ablated'
    if ablation_type.lower() == 'noise':
        n_units_ablated_key = 'scale of per-unit noise'
    dropout_key = 'dropout fraction'
    rep_key = 'repetition'
    # Initialize ablation_data keys if it's empty
    if not ablation_data:
        ablation_data_keys = [shuff_key, loss_key, accuracy_key,
                              n_units_ablated_key, dropout_key, rep_key]
        for key in ablation_data_keys:
            ablation_data[key] = []

        # List of models in directory
    files = [f for f in os.listdir(path_models_and_targets) \
            if os.path.isfile(os.path.join(path_models_and_targets, f)) \
            and 'model' in f]
    # Loop through and analyze each model file
    for model_name in files:
        print('Analyzing file: ' + model_name)
        # Shuffled labels. Find file with matching model parameter file
        # name substring
        path_shuffled_targets_file = path_models_and_targets \
            + model_name[0: model_name.find('epochs')] + 'targets.pt'
        # Load shuffled labels
        data_loader.dataset.targets = torch.load(path_shuffled_targets_file)
        n_units = len(model.ablation_list)
        # If ablating to zero
        if ablation_type.lower() == 'zero':
            if ablation_steps is None:
                ablation_steps = n_units
            # Ablation counts on which model should be tested
            ablation_test_counts = np.linspace(0, n_units,
                                               ablation_steps).round()
        # If injecting noise
        elif ablation_type.lower() == 'noise':
            if ablation_steps is None:
                ablation_steps = 20
            ablation_test_counts = torch.logspace(
                noise_scale[0], noise_scale[1], ablation_steps)
        # Create values for ablation_curves
        shuffle_fraction = \
            float(model_name[model_name.find('shuff') + 5:
                                  model_name.find('shuff') + 8])
        dropout_fraction = \
            float(model_name[model_name.find('dropout') + 7:
                                  model_name.find('dropout') + 10])
        # Repeat ablation process for repetitions
        for rep_n in range(n_repetitions):
            # Load Model
            model.load_state_dict(torch.load(path_models_and_targets +
                                             model_name, map_location=device))
            print('Repetition ' + str(rep_n + 1) + ' out of ' \
                  + str(n_repetitions))
            # If we are zeroing
            if ablation_type.lower() == 'zero':
                # Flag to continue ablating
                ablation_flag = True
                # Count of units ablated
                n_units_ablated = 0
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
            # If injecting noise
            elif ablation_type.lower() == 'noise':
                # Compute (or load) activations of every unit for every example
                # Find file containing standard deviations of each unit's
                # activation across the training data. If file doesn't
                # exist, obtain data and create it.
                unit_data_filename = \
                    model_name[0: model_name.find('epochs')]
                path_activation_data = path_activations + \
                                    unit_data_filename + 'activations.pt'
                path_unit_sd_data = path_activations + \
                                    unit_data_filename + 'sd.pt'
                if os.path.exists(path_unit_sd_data):
                    model.sd = torch.load(path_unit_sd_data)
                else:
                    loss, accuracy = [], []
                    model.store_activations = True
                    if model.sd_scale and model.store_activations:
                        warnings.warn('Unit activations are being stored '
                                      'while noise is being injected. Set '
                                      'model.sd_scale to zero to avoid '
                                      'storing noise-injected unit '
                                      'activations.')
                    # Test on full training data
                    test(model, data_loader, criterion, loss, accuracy,
                         device=device)
                    # Convert activation data to tensor
                    model.activations = torch.tensor(model.activations)
                    # Save activation data
                    torch.save(model.activations, path_activation_data)
                    # Compute standard deviations
                    model.sd = model.activations.std(1)
                    # Save standard deviations of activations
                    torch.save(model.sd, path_unit_sd_data)
                    model.store_activations = False
                    model.activations = [[], []]
                # Loop through scales of noise
                for current_noise_scale in ablation_test_counts:
                    model.sd_scale = current_noise_scale
                    loss, accuracy = [], []
                    print('Noise scale: ' + str(current_noise_scale))
                    # Test
                    test(model, data_loader, criterion, loss,
                         accuracy, data_fraction=data_fraction,
                         device=device)
                    # Add values to ablation_data
                    ablation_data[shuff_key].append(
                        shuffle_fraction)
                    ablation_data[loss_key].append(loss[0])
                    ablation_data[accuracy_key].append(accuracy[0].item())
                    ablation_data[n_units_ablated_key].append(
                        current_noise_scale.item())
                    ablation_data[dropout_key].append(dropout_fraction)
                    ablation_data[rep_key].append(rep_n)
                    # Print accuracy
                    print('Accuracy: ' + str(accuracy[0].item()))
                    # Reset noise scaling to 0
                    model.sd_scale = 0
                # Loop through ablation_test_counts, adding scaled noise
    return ablation_data

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    From: discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/4
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = math.floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1)
                                             ) - 1 )/ stride) + 1)
    w = math.floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1)
                                             ) - 1 )/ stride) + 1)
    return h, w
