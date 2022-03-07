#! /usr/bin/env python

"""
Load a model to predict if figure 8 ghosts appear in WFC3 images. Also contains
utility functions for producing saliency maps.

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

%run utils.py

or

from utils import <functions>
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Classifier(nn.Module):
    def __init__(self,
                 filters = [1, 32, 32, 8], # most people use powers of 2
                 neurons = [16, 8, 2],
                 sub_array_size = 256,   # image size (28x28)
                 k = 3,                 # kernel size (3x3)
                 pool = 2,              # pooling size (2x2)
                 pad = 1):

        super(Classifier, self).__init__()

        # The Rectified Linear Unit (ReLU)
        self.relu = nn.ReLU()

        # Max Pool
        self.mp = nn.MaxPool2d(pool, return_indices=False)


        # Flattens the feature map to a 1D array
        self.flatten = Flatten()

        # ---- CONVOLUTION  ----
        self.conv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=k, padding=pad)
        self.conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=k+2, padding=pad+1)
        self.conv3 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=k, padding=pad)


        self.bn1 = nn.BatchNorm2d(filters[2])
        self.bn2 = nn.BatchNorm2d(filters[3])


        # ---- FULLY CONNECTED ----
        neurons_flat = filters[-1] * (sub_array_size // (self.mp.kernel_size ** 3)) ** 2 # only works with k=3, pool=1


        self.fc1 = nn.Linear(neurons_flat, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])


        self.dropout1 = nn.Dropout(0.2)



    def forward(self,x):

        # Convolutional Layer 1
        x = self.conv1(x)
        x = self.mp(x)
        x = self.relu(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Convolutional Layer 3
        x = self.conv3(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.bn2(x)


        # Flatten Layer
        x = self.flatten(x)
        x = self.dropout1(x)

        # Fully Connected 1
        x = self.fc1(x)
        x = self.relu(x)

        # Fully Connected 2
        x = self.fc2(x)
        x = self.relu(x)

        # Fully Connected 3
        x = self.fc3(x)

        return x


def load_wfc3_uvis_figure8_model(model_path='wfc3_uvis_figure8_model_c.torch'):
    """
    Construct the model C six layer CNN.  Freeze all layers

    Load the weights and biases trained of WFC3 figure 8 ghosts and change model
    to eval mode (turns off gradients).

    Parameters
    ----------
    model_path : string
        Path to saved model.

    Returns
    -------
    model : Classifier
        Transfer learned CNN for WFC3 figure 8 ghosts.
    """

    # Construct network without trained weights
    model = Classifier()

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Load pretrained weight and biases and change to eval mode
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model



def saliency_map(model, image, plot=True):

    """Plot a subframe and saliency map the model produces.

    Parameters
    ----------
    model : nn.Module
        Trained CNN after a given number of epochs.

    image : numpy array
        A processed UVIS image.

    plot : boolean
        If True, plot image and saliency map.

    Returns
    -------
    sal_map : array like
        The saliency map produced by the model from the input image.

    """

    # Rename image and label
    X = torch.Tensor(image.reshape(1,1,256,256))

    # Change model to evaluation mode and activate gradient
    model.eval()
    X.requires_grad_()

    # Evaluate image and perform backwards propogation
    scores = model(X)
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]
    score_max.backward()

    # Calculate saliency map
    saliency, _ = torch.max(X.grad.data.abs(),dim=1)
    sal_map = saliency[0]

    softmax = torch.nn.Softmax(dim=1)

    # Plot
    if plot:

        # Display probabilities
        prob = softmax(scores).detach().numpy().flatten()
        print ('Null Probability: {:.4f}'.format(prob[0]))
        print ('Figure 8 Probability: {:.4f}'.format(prob[1]))
        print ('Prediction: {}'.format(score_max_index))

        # Plot image and saliency map
        fig, axs = plt.subplots(1, 2, figsize=[24,12])
        axs[0].set_title('Input Image', fontsize=20)
        axs[0].imshow(X[0, 0].detach().numpy(), cmap='gray', origin='lower')
        axs[0].tick_params(axis='both', which='major', labelsize=20)

        axs[1].set_title('Saliency Map', fontsize=20)
        axs[1].imshow(sal_map, cmap=plt.cm.hot, origin='lower')
        axs[1].tick_params(axis='both', which='major', labelsize=20)

        plt.show()

    return sal_map.numpy()
