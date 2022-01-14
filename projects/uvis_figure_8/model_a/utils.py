#! /usr/bin/env python

"""
Load a model to predict if figure 8 ghosts appear in WFC3 images. Also contains
utility functions for normalizing images to match ImageNet statistics and
producing saliency maps.

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
import torchvision
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image):
    """
    Normalize an image to match ImageNet statistics.

    GoogLeNet was trained using the ImageNet dataset so samples need to be
    scaled accordingly for classification.

    We min-max scale the image, concatenate three copies to act as RGB channels,
    center crop to a 224x224, and normalize to the statistics in preprocess.

    Parameters
    ----------
    image : numpy.array
        A 256x256 WFC3 image.

    image_processed : torch.Tensor
        The ImageNet scaled 3x224x224 WFC3 image.
    """

    image = image.reshape(1,256,256)
    image_scale = (image - image.min()) / (image.max() - image.min())
    image_3 = np.concatenate((image_scale, image_scale, image_scale))
    image_rgb = np.transpose(image_3, axes=(1,2,0))
    input_tensor = preprocess(image_rgb)
    image_processed = input_tensor.reshape(1,3,224,224).float()

    return image_processed

def load_wfc3_uvis_figure8_model(model_path='googlenet_fig8_deep.torch'):
    """
    Load model pretrained by GoogLeNet and retrained by DeepWFC3.

    First load the pretrained model and freeze all layers. Append two 1024
    neuron layers with dropout rates of 0.5 and ReLU activation. Additionally
    append one 2 neuron layer with a dropout rate of 0.2. Unfreeze the fully connected layers.

    Load the weights and biases trained of WFC3 figure 8 ghosts and change model
    to eval mode (turns off gradients).

    Parameters
    ----------
    model_path : string
        Path to saved model.

    Returns
    -------
    model : torchvision.models.googlenet.GoogLeNet
        Transfer learned GoogLeNet for WFC3 figure 8 ghosts.
    """

    # Load GoogLeNet from torchvision
    model = torchvision.models.googlenet(pretrained=True)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Append fully connected layers
    num_input_features = model.fc.in_features
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Sequential(
        nn.Linear(num_input_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024,2)
    )

    # Unfreeze fully connected layers
    for param in model.fc.parameters():
        param.requires_grad = True

    # Load pretrained weight and biases and change to eval mode
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

softmax = torch.nn.Softmax(dim=1)

def saliency_map(model, image, plot=True):

    """Plot a subframe and saliency map the model produces.

    Parameters
    ----------
    model : nn.Module
        Trained CNN after a given number of epochs.

    image : numpy array
        A processed IR subframe.

    plot : boolean
        If True, plot image and saliency map.

    Returns
    -------
    sal_map : array like
        The saliency map produced by the model from the input image.

    """

    # Rename image and label
    X = torch.Tensor(image.reshape(1,3,224,224))

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
