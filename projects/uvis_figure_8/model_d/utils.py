#! /usr/bin/env python

"""
Load a model to predict if figure 8 ghosts appear in WFC3 images. Also building the model, contains 
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
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# convenient functions
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# define functions and build model

class Classifier(nn.Module):
    def __init__(self, 
                 filters = [1, 16, 32, 64], # most people use powers of 2
                 neurons = [32,64,16,2],   # neurons of fully connected layer
                 sub_array_size = 256,   # image size (256x256)
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
        self.conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=k, padding=pad)
        self.conv3 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=k, padding=pad)
        
        dropout = 0.5
        self.dropout1 = nn.Dropout(.5*dropout)
        self.dropout2 = nn.Dropout(dropout)

        # ---- FULLY CONNECTED ----
        neurons_flat = filters[-1] * (sub_array_size // pool**3)
        self.fc1 = nn.Linear(neurons_flat, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])
        self.fc4 = nn.Linear(neurons[2], neurons[3])
        
        # ---- Batch Normalization
        self.bn3 = nn.BatchNorm2d(filters[3])
        
    def forward(self, x):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = self.relu(x)        
        x = self.mp(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = self.relu(x)         
        x = self.mp(x)

        # Convolutional Layer 3
        x = self.conv3(x)
        x = self.relu(x)          
        x = self.mp(x)
      
        x = self.bn3(x)
        
        # Flatten Layer
        x = self.dropout1(x)
        x = self.flatten(x)
        
        # Fully Connected 1
        x = self.fc1(x)
        x = self.relu(x)

        # Fully Connected 2
        x = self.dropout2(x)
        x = self.fc2(x)      
        x = self.relu(x)
        
        
        # Fully Connected 3
        x = self.fc3(x)   
        x = self.relu(x)
        
        x = self.fc4(x)  
        
        return x

def load_wfc3_uvis_figure8_model(model_path='wfc3_uvis_figure8_model_d.torch'):
    """
    Load model pretrained by transfer learned model by DeepWFC3.

    
    Parameters
    ----------
    model_path : string
        Path to saved model.

    Returns
    -------
    model : return model for WFC3 figure 8 ghosts.
    """

    # initialize model
    model = Classifier()

    # define loss function
    distance = nn.CrossEntropyLoss()

    # define optimizer : automatically determines learning rate for you
    optimizer = torch.optim.Adam(model.parameters(),  weight_decay=1e-5)

    cnt = 0

    params = model.state_dict()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # freezing the convolution layers
    for param in model.parameters():
        if cnt < 6:
            param.requires_grad = False

        cnt += 1

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path ))
    model.eval()

    return model


def saliency_map(model, image, label, index, name):
    
    """Plot a subframe and saliency map the model produces.
    
    Parameters
    ----------
    model : nn.Module
        Trained CNN after a given number of epochs. 
        
    image : numpy array
        A processed IR subframe.
    
    label : float, integer, string
        Label corresponding to the image.
        
    index : integer
        Index corresponding to the image and label
        
    Returns
    -------
    sal_map : array like
        The saliency map produced by the model from the input image.
        
    """
    
    # Rename image and label
    X = torch.Tensor(image.reshape(1,1,256,256))
    Y = label
    
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
    
    # Plot image and saliency map
    fig, axs = plt.subplots(1, 2, figsize=[16,8])
    axs[0].set_title(name)
    axs[0].imshow(X[0, 0].detach().numpy(), cmap='Greys', origin='lower')
    axs[1].set_title('Saliency Map for {}'.format(name))
    axs[1].imshow(sal_map, cmap=plt.cm.hot, origin='lower')
    
    if Y == float:
        print ('True Label: {}'.format(int(Y)))
    else:
        print ('True Label: {}'.format(Y))
        
    print ('Prediction: {}'.format(score_max_index))
    
    plt.show()
    
    return sal_map