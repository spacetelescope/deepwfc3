# imports:
import torch
from torch import nn

class Model(nn.Module):
    ''' Define the convolutional neural network (CNN) model architecture to 
    train on.
    
    The CNN defined in this class has four convolutional layers, and two fully 
    connected layers that have the following characteristics:
        - Convolutional layer 1: 32 filters, (2,2) max pooling, ReLu activation
        - Convolutional layer 2: 64 filters, (4,4) max pooling, ReLu activation
        - Convolutional layer 3: 128 filters, (4,4) max pooling, ReLu activation
        - Convolutional layer 4: 256 filters, (4,4) max pooling, ReLu activation
        - Fully connected layer 1: 64 neurons, ReLu activation
        - Fully connected layer 2: 2 neurons

    The model uses a kernel size of 3, and uses 1 row of padding in the 
    convolutional layers.
    Parameters
    ----------
    sub_array_size : int
        The size of the images that are being input into the model.
    Returns
    -------
    x : Torch tensor
        The model prediction for data.
    '''
    def __init__(self, sub_array_size, filters = [1, 32, 64, 128, 256],  
                 neurons = [64, 2],         # neurons of fully connected layer 
                 k = 3,                     # kernel size (3x3)
                 pool = 4,                  # pooling size (4x4)
                 pool2 = 2,
                 pad = 1): 
        super(Model, self).__init__()

        # ReLU function:
        self.relu = nn.ReLU()

        # Max pooling:
        self.mp4 = nn.MaxPool2d(pool, return_indices=False)
        self.mp2 = nn.MaxPool2d(pool2, return_indices=False)

        # flatten feature map:
        self.flatten  = nn.Flatten(start_dim=1)

        ### CONVOLUTIONAL LAYERS ###
        self.conv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], 
                               kernel_size=k, padding=pad)

        self.conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], 
                               kernel_size=k, padding=pad)
        
        self.conv3 = nn.Conv2d(in_channels = filters[2], out_channels = 
                               filters[3], kernel_size=k, padding=pad)

        self.conv4 = nn.Conv2d(in_channels = filters[3], out_channels = 
                               filters[4], kernel_size=k, padding=pad)
        

        num_pool4 = len(filters) - 2 
        # -2 because 1 is 2x2 pooling
        pool_size = (sub_array_size // 2) // (pool ** num_pool4)
        # Size of image after 2x2 pooling = sub_array_size // 2 **[n_2x2_pools]
        # Size of image after all pooling = 2x2_pool_size // 4 **[n_4x4_pools]

        neurons_flat = filters[-1] * (pool_size**2)
        
        ### FULLY CONNECTED LAYERS ###        
        self.fc1 = nn.Linear(neurons_flat, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
    
    def forward(self,x):

        # Convolutional Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp2(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp4(x)

        # Convolutional Layer 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp4(x)

        # Convolutional Layer 4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.mp4(x)
        
        # Flatten Layer
        x = self.flatten(x)

        # Fully Connected 1
        x = self.fc1(x)
        x = self.relu(x)
        
        # Fully Connected 2
        x = self.fc2(x)  

        return x