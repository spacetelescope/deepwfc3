#! /usr/bin/env python

"""
Load a convolutional neural network (CNN) trained on synthetic figure-8 ghosts to predict if figure-8 ghosts appear in WFC3 images.

Authors
-------

Members of DeepWFC3 2022

    Frederick Dauphin
    Mireia Montes
    Nilufar Easmin
    Varun Bajaj
    Peter McCullough

Use
---

This script is intened to be used in conjunction with a jupyter notebook.

%run model_syn_utils.py

or

from model_syn_utils import <functions>
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torchvision import transforms

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model_Synthetic(nn.Module):
    def __init__(self,
                 filters = [1, 16, 32],
                 neurons = [512, 64, 2],
                 sub_array_size = 256,
                 k = 5,
                 pool = 4,
                 pad = 2,
                 dropout = 0.2):

        super(Model_Synthetic, self).__init__()

        #MetaData and Functions
        self.sub_array_size = sub_array_size
        self.filters = filters
        self.pool = pool
        self.flat_neurons = filters[-1] * (sub_array_size // (pool ** 2)) ** 2

        self.mp = nn.MaxPool2d(pool)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.logsm = nn.LogSoftmax(dim=1)

        #Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=k, padding=pad)
        self.conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=k, padding=pad)

        #Dropout Rate
        self.dropout1 = nn.Dropout(2 * dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout / 2)

        #Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_neurons, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])


    def forward(self, x):

        #Conv Layer 1 --> Activate --> Max Pool (256 --> 64)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)

        #Conv Layer 2 --> Activate --> Max Pool (64 --> 16)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)

        #Dropout 0.4 Neurons and Flatten (32 * 16 * 16 neurons)
        x = self.flatten(x)
        x = self.dropout1(x)

        #FC Layer 1 (Flatten --> 512)
        x = self.fc1(x)
        x = self.relu(x)

        #Dropuut 0.2 Neurons, FC Layer 2 (512 --> 64)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)

        #Dropuut 0.1 Neurons, FC Layer 3 (64 --> 2)
        x = self.dropout3(x)
        x = self.fc3(x)

        return x

def load_wfc3_fig8_model_syn(model_path='wfc3_fig8_model_syn.torch'):
    """
    Load model trained on synthetic figure-8 ghost images and change model to eval mode (turns off gradients).

    Parameters
    ----------
    model_path : string
        Path to saved model.

    Returns
    -------
    model : Model_Synthetic
        Model with pretrained weights.
    """

    # Load pretrained weight and biases and change to eval mode
    model = Model_Synthetic()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
