#! /usr/bin/env python

""" Build a Convolutional Neural Network (CNN) that classifies blob images in
    the WFC3 IR Channel.

This script:

    1. Processes IR images. 

        The data processing and augmentation pipeline is as follows:

        - Use vmin/vmax from zscale function to determine clip min/max values for data.

        - Change border and nan values to median.

        - Normalize data to N(0, 1), where blobs are positive values, background is near 0, and 
        random noise are negative values.

        - Grab random 256x256 subframes from a 1024x1024 image. 
        The size of the subframe significantly increases computation efficiency.

        - Add random noise (N(0, 0.75)), rotation, and flipping to subframes to diversify data set.

    2. Produces training, validation, and test sets of IR blob subframes.

        With the data processing pipeline, we can generate and save training, validation, and test sets,
        containinng hundreds to thousands of subframes from 1024x1024 blob images. 
        We can even superimpose blobs onto non blob subframes to increase blob data set diversity,
        which increases the chances that the model truly learns what a blob is.

    3. Trains a CNN to classify if blobs are in a subframe.
        
        We train a convolutional neural network with the following initial hyperparameters:
        
        - 2 convolutional layers (1 filter to 8 filters to 16 filters)
        - 2 fully connected layers (16 * 64 * 64 neurons to 128 neurons to 2 neurons)
        - 5x5 kernel
        - 2x2 max pooling at the end of each convolutional layer
        - 2 padding on each feature map
            -- This ensures the feature maps don't shrink after a convolution, but before a max pool
        - 15% and 30 % dropout regularization
        - Cross Entropy Loss
        - Adam optimizer
        - Batch size of 100
        - 5 epochs

    4. Evaluates the model's performance.
    
        We use loss, accuracy, and confusion matrices as metrics for evaluating the model's performance.
        In addition, we check incorrect images and plot saliency maps to further investigate how and why
        the model makes certain predictions.
    

Definitions
-----------

Blob Image : classification 1
    An IR image with blobs, which are dark circular distortions on the image. 
    See (insert TIR/ISR) for a more complete definition.

Non Blob Image : classification 0
    An IR image without blobs on the image, which is just background noise.
    
Median Stack :
    A median stack of a set of blob images, usually from the past 30 days.

Blob Difference Image :
    An image produced by taking the difference between a median stack and a current blob image.

Subframe : 
    A 256x256 cut out from the 1024x1024 IR image. 
    This size ensures that if we were using subframes of the blob image or median stack
    for our blob samples, the probability of choosing a random subframe and there being a blob
    is nearly 1.


Authors
-------

Members of DeepWFC3 2021

    Frederick Dauphin
    Jennifer Medina
    Peter McCullough

Use
---

This script is intened to be used in conjunction with a jupyter notebook (TBD).

%run wfc3_ir_blob_class_utils.py

"""
# Imports and define blob dictionary used for superimposing
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from ginga.util.zscale import zscale
from astropy.io import fits
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import seaborn as sns

BLOB_DICT = np.load('blob_dict.npy', allow_pickle=True).item()

################################################################################################################################
####################################################### DATA PROCESSING #######################################################
################################################################################################################################

def plot_pre_processed(image_path, show_hist=True, stack=0):

    """Plot preprocessed IR image.

    The shape of non blob images and median stacks is (3, 1024, 1024) so we
    only keep the first extension as data. The shape of the blob images is
    (1024, 1024) so this problem does not arise.

    Parameters
    ----------
    image_path : string
        The path for the image that will be displayed. This path must be a .fits
        file.

    show_hist : boolean
        If True, display the histogram of the finite pixel values in the image.
        If False, do not display the histogram.

    stack : integer
        Use 0 for non blob and blob images. Use 1 for median stacks. The integer
        will give the correct extension for each image type.

    Returns
    -------
    data : numpy array
        The preprocessed IR image, which may contain nonfinite values,
        such as NaN and +/- Inf.

    """

    # Retrieve data
    data = fits.getdata(image_path, stack)

    # Use the first extension of the image for non blob images and median stacks
    if data.ndim == 3:
        data = data[0]

    # Use vmin/vmax for image display
    in_vmin, in_vmax = zscale(data)

    # Plot full IR image
    plt.figure(figsize=[10,10])
    im = plt.imshow(data, vmin=in_vmin, vmax=in_vmax, origin='lower', cmap='Greys')
    cbar = plt.colorbar(im, fraction=0.0459, pad=0.04)

    plt.title(os.path.basename(image_path + ' preprocessed'), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    plt.show()

    # Display distribution of pixel values to ensure normalization
    if show_hist:
        plt.title('Pixel Distribution')
        plt.hist(data.flatten()[np.isfinite(data.flatten())], bins=50)
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.show()

    return data

def scale_data(data, factor=0.5):

    """Clip data using a fraction of vmin/vmax as boundary min/max pixel values.
       Change NaN and image border pixels to the median.
       Normalize data to N(mean=0, var=1).

    We determine our clipping min and max by using a fraction of vmin and
    vmax. It can be empirically shown that 0.3, 0.5 and 0.6 of vmin/vmax
    contains approximately +/- 2, 3, and 4 standard deviations of the data
    centered around the mean. The borders of the IR images mostly contains
    nonfinite values so changing their pixel values to the median is beneficial
    for data processing.

    Parameters
    ----------
    data : numpy array
        The preprocessed IR image.

    factor : float
        The fraction of vmin/vmax to use for clipping. Factors of 0.3, 0.5, and
        0.6 keep approximately 95%, 99.7%, and 99.994% of the data centered
        around the mean. The lower the factor, the stronger the clipping effect
        will be.

    Returns
    -------
    data_normal : numpy array
        The IR image with all NaN and border pixel values changed to the median,
        all pixel values outside of clip min/max clipped, and normalized to a
        standard normal distribution.

    """

    # Find median, vmin, and vmax pixel values
    median = np.nanmedian(data)
    vmin, vmax = zscale(data)

    # Find clip min/max values from vmin/vmax
    clip_min, clip_max = factor * vmin, factor * vmax

    # Find indices that are outside clip min/max values
    clip_ind = np.where((data < clip_min) | (data > clip_max))

    # Make a copy of data to change pixel values
    data_clip = np.copy(data)

    # Change nans to median
    nan_ind = np.where(np.isnan(data_clip))
    for row, col in zip(nan_ind[0], nan_ind[1]):
        data_clip[row, col] = median

    for row, col in zip(clip_ind[0], clip_ind[1]):
      # Change borders to median
        if (row < 10 or row > 1000) or (col < 10 or col > 1000):
            data_clip[row, col] = median
        else:
          # Change values outside of clip boundaries to clip values
            if data_clip[row, col] < clip_min:
                data_clip[row, col] = clip_min
            elif data_clip[row, col] > clip_max:
                data_clip[row, col] = clip_max

    # Check all inf/nan values have been changed
    check_mask = (np.abs(data_clip) == np.inf) | np.isnan(data_clip)
    if list(data_clip[check_mask]):
        print ('Nonfinite values still remian before normalization.')
    else:
        print ('Nonfinite values before normalizing have been changed.')

    # Normalize data
    data_normal = (data_clip - np.mean(data_clip)) / np.std(data_clip)

    return data_normal

def plot_scaled_blobs(image_path, show_hist=True, stack=0, factor=0.5):

    """Plot processed IR image.

    The shape of non blob images and median stacks is (3, 1024, 1024) so we only
    keep the first extension as data. The shape of the blob images is
    (1024, 1024) so this problem does not arise.

    Parameters
    ----------
    image_path : string
        The path for the image that will be displayed. This path must be a .fits
        file.

    show_hist : boolean
        If True, display the histogram of the finite pixel values in the image.
        If False, do not display the histogram.

    stack : integer
        Use 0 for non blob and blob images. Use 1 for median stacks. The integer
        will give the correct extension for each image type.

    factor : float
        The fraction of vmin/vmax to use for clipping.

    Returns
    -------
    data_normal : numpy array
        The IR image with all NaN and border pixel values changed to the median,
        all pixel values outside of clip min/max clipped, and normalized to a
        standard normal distribution.

    """

    # Retreive data
    data = fits.getdata(image_path, stack)

    # Use the first extension of the image for blob images and median stacks
    if data.ndim == 3:
        data = data[0]

    # Clip and normalize image
    data_normal = scale_data(data, factor)

    # Plot processed IR image
    plt.figure(figsize=[10,10])
    im = plt.imshow(data_normal, origin='lower', cmap='Greys')
    cbar = plt.colorbar(im, fraction=0.0459, pad=0.04)

    plt.title(os.path.basename(image_path + ' processed'), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('unit', rotation=270, fontsize=20)

    plt.show()

    # Display distribution of pixel values to ensure normalization
    if show_hist:
        plt.title('Pixel Distribution (Normalized)')
        plt.hist(data_normal.flatten(), bins=50)
        plt.xlabel('Pixel Values (Normalized)')
        plt.ylabel('Frequency')
        plt.show()

    return data_normal

def grab_full_subframes(data_normal):

    """Cut IR image into 16 unique subframes.

    The final test for the CNN will be classifying 16 unique subframes from a
    non blob image and a blob image.

    Parameters
    ----------
    data_normal : numpy array
        The IR image with all NaN and border pixel values changed to the median,
        all pixel values outside of clip min/max clipped, and normalized to a
        standard normal distribution.

    Returns
    -------
    subframes : numpy array
        16 unique subframes of the processed IR image.

    """
    
    subframe_lst = []
    length = data_normal.shape[0]
    subframe_length = length // 4

    # Loop through 16 times and get unique subframes
    for row in range (0, length, subframe_length):
        for col in range (0, length, subframe_length):
            row_end = row + subframe_length
            col_end = col + subframe_length
            subframe = data_normal[row:row_end, col:col_end]
            subframe_lst.append(subframe)

    subframes = np.array(subframe_lst)

    return subframes

def grab_random_subframe(data_normal, 
                         ix=None, 
                         iy=None, 
                         subsize=256, 
                         sets=[False, False]):

    """Grab a random subframe from a processed IR image.

    The training, validation, and test sets will be generated by grabbing random
    subframes from a processed IR image and augmenting the subframes to increase
    diversity of the sets. Use only the left 75% of the image for training and
    the right 25% for validation.

    Parameters
    ----------
    data_normal : numpy array
        The IR image with all NaN and border pixel values changed to the median,
        all pixel values outside of clip min/max clipped, and normalized to a
        standard normal distribution.

    ix : integer
        Index on the x axis. If None, grad random index.

    iy : integer
        Index on the y axis. If None, grad random index.

    subsize : integer
        The size of the subframe.

    sets : list of booleans
        The indices correspoond to training and validation sets, respectively.
        If either are True, the random draw will be restricted to either the
        left or right side of the image. If both are False, the random draw will
        be anywhere on the image. Do not choose both to be True.

    Returns
    -------
    subframe_random : numpy array
        A random subframe of the processed IR image.

    """
    
    training, validation = sets

    # Only use the left side of the image for making training sets
    if training:
        data_normal = data_normal[:, :subsize * 3]
        
    ny, nx = data_normal.shape
    
    # Choose x index
    if ix is None:
      # Only use the right side of the image for making validation sets
        if validation:
            ix = int(subsize * 3.5)
        else:
            ix = np.random.randint(subsize // 2, nx - subsize // 2)

    # Choose y index
    if iy is None:
         iy = np.random.randint(subsize // 2, ny - subsize // 2)

    # Set subframe indices
    xstart, xstop = ix - subsize // 2, ix + subsize // 2
    ystart, ystop = iy - subsize // 2, iy + subsize // 2

    # Use indices to retrieve subframe
    subframe_random = data_normal[ystart:ystop, xstart:xstop]

    return subframe_random

def flip(subframe_random):

    """Randomly flip subframe.

    Parameters
    ----------
    subframe_random : numpy array
        A random subframe from a processed IR image.

    Returns
    -------
    subframe_flip : numpy array
        The random subframe flipped across a random axis.

    """
    
    # Choose a random index
    num = np.random.randint(0,4)

    # Randomly flip subframe
    if num == 0:
        subframe_flip = subframe_random
    elif num == 1:
        subframe_flip = np.flip(subframe_random)
    elif num == 2:
        subframe_flip = np.flip(subframe_random, 0)
    else:
        subframe_flip = np.flip(subframe_random, 1)

    return subframe_flip

def add_blobs(subframe_non_blob):

    """Add random blobs from the blob dictionary to a non blob image.

    Superimposing blobs will diversify the training set in blob numbers, size,
    and intensity. This diversity allows the CNN to learn what a "blob" actually
    is versus learning what a "blob image" is.

    Parameters
    ----------
    subframe_non_blob : numpy array
        A non blob subframe to superimpose blobs on.

    Returns
    -------
    subframe_blob : numpy array
         A non blob subframe with a random number of blobs superimposed with
         varying intensity, position, size, and number.

    """
    
    image = subframe_non_blob.copy()
    image_length = image.shape[0]
    lst_num_of_blob = np.random.randint(1, 6)

    # Superimpose a random number of blobs from one to five
    for i in range (lst_num_of_blob):

        # Randomly select a blob
        rand_blob_ind = str(np.random.randint(1, 28))
        blob = BLOB_DICT[rand_blob_ind]
        size = blob.shape[0]

        # Randomly change blob intensity and rotate
        factor = np.random.uniform(0.5, 1.5)
        rot_blob = np.random.randint(0, 4)
        blob = flip(np.rot90(factor * blob, rot_blob))

        # Randomly choose blob position
        # Which could be at most half off the frame on either side
        x, y = np.random.randint(0, image_length - size//2, 2)
        x_edge = image_length - x
        y_edge = image_length - y

        # Change blob shape if random position is off the frame
        if x_edge < size:
            blob = blob[:, :x_edge]
        if y_edge < size:
            blob = blob[:y_edge, :]

        # Superimpose blob
        image[y:y+blob.shape[0], x:x+blob.shape[1]] = blob

    # Normalize superimposed image
    subframe_blob = (image - image.mean()) / image.std()

    return subframe_blob

def make_dataset(image_path, 
                 itertimes, 
                 stack=0, 
                 factor=0.5, 
                 plot=False, 
                 superimpose=False, 
                 sets=[False, False]):

    """Generate a dataset of random subframes with noise, flipping, and rotation.

    Parameters
    ----------
    image_path : string
        The path for the image that will be displayed. This path must be a .fits
        file.

    itertimes : integer
        The number of random subframes in the dataset.

    stack : integer
        Use 0 for non blob and blob images. Use 1 for median stacks. The integer
        will give the correct extension for each image type.

    factor : float
        The fraction of vmin/vmax to use for clipping.

    plot : boolean
        If True, display the histogram of the finite pixel values in the image.
        If False, do not display the histogram.

    superimpose : boolean
        If True, superimpose blobs onto subframes.
        If False, do not superimpose blobs.

    sets : list of booleans
        The indices correspoond to training and validation sets, respectively.
        If either are True, the random draw will be restricted to either the
        left or right side of the image. If both are False, the random draw will
        be anywhere on the image. Do not choose both to be True.

    Returns
    -------
    dataset : numpy array
         An array with itertimes amount of subframes with random position,
         rotation, noise, flip, and superimposed blobs if necessary.

    """

    # Retrieve data
    data = fits.getdata(image_path, stack)

    # Use the first extension of the image for blob images and median stacks
    if data.ndim == 3:
        data = data[0]

    # Clip and normalize image
    data_normal = scale_data(data, factor)

    # Plot histogram of pixel values from normalized data
    if plot:
        plt.title('Scaled Pixel Distribution')
        plt.hist(data_normal.flatten())
        plt.xlabel('Scaled Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    dataset = []

    # Make itertimes amount of random subframes
    for n in range(0, itertimes):

        # Grab a random subframe from image
        subframe_random = grab_random_subframe(data_normal, sets=sets)

        # Superimpose blobs if necessary
        if superimpose:
            subframe_random = add_blobs(subframe_random)

        # Randomly rotate subframe
        rchoice = np.random.randint(0,4)
        subframe_rot = np.rot90(subframe_random, rchoice)

        # Randomly add noise to subframe
        noise = np.random.normal(0, .75, size=np.shape(subframe_rot))
        subframe_noise = subframe_rot + noise

        # Randomly flip subframe
        subframe_flip = flip(subframe_noise)

        dataset.append(subframe_flip)
    
    dataset = np.array(dataset)

    return dataset

def save_generated_dataset(lst_generate,
                            save_path,
                            blob=False,
                            itertimes=1000,
                            stack=0,
                            factor=0.5,
                            plot=False,
                            superimpose=False,
                            sets=[False, False]):

    """Save a generated dataset.

    Parameters
    ----------
    lst_generate : list of strings
        List of the file names to generate datasets from.

    save_path : string
        The path where the dataset is saved.

    blob : boolean
        If True, corresponding labels are 1.
        If False, corresponding labels are 0.

    itertimes : integer
        The number of random subframes in the dataset.

    stack : integer
        Use 0 for non blob and blob images. Use 1 for median stacks. The integer
        will give the correct extension for each image type.

    factor : float
        The fraction of vmin/vmax to use for clipping.

    plot : boolean
        If True, display the histogram of the finite pixel values in the image.
        If False, do not display the histogram.

    superimpose : boolean
        If True, superimpose blobs onto subframes.
        If False, do not superimpose blobs.

    set : list of booleans
        The indices correspoond to training and validation sets, respectively.
        If either are True, the random draw will be restricted to either the
        left or right side of the image. If both are False, the random draw will
        be anywhere on the image. Do not choose both to be True.

    Returns
    -------
    save_image_set : numpy array
        The image set generated from lst_generate.
        
    save_labels : numpy array
        Labels corresponding to the image set.

    """
    
    # Loop through each file to generate image sets
    image_set = []
    for file in lst_generate:
        image_set.append(make_dataset(file, 
                                      itertimes, 
                                      stack, 
                                      factor, 
                                      plot, 
                                      superimpose, 
                                      sets)
                        )
    
    # Concatenate all image sets to one array 
    # Shape = (len(lst_generate) * itertimes, 256, 256)
    save_image_set = np.concatenate(image_set)
    
    # Set labels for blobs (1) or non blobs (0)
    if blob:
        save_labels = np.ones(shape=save_image_set.shape[0])
    else:
        save_labels = np.zeros(shape=save_image_set.shape[0])
        
    # Save image set and labels as a .npz file
    np.savez_compressed(save_path, 
                        image_set=save_image_set, 
                        labels=save_labels)

    print ('Dataset of size {} saved to {}'.format(save_labels.shape[0], 
                                                   save_path)
          )

    return save_image_set, save_labels

################################################################################################################################
####################################################### MODELING #######################################################
################################################################################################################################

# At this point, you should have training/validation sets of size (itertimes, 256, 256)
# And training/validation labels of size (itertimes).

def generate_test_data(image_set, labels, num):
    
    """Randomly generate test set from previously generated image set and labels.
    
    This function keeps track of previously sampled data to make sure each subframe 
    in the test set is unique. If num is greater than the shape of the image set, 
    this function will crash because it has sampled every subframe in the image set.

    Parameters
    ----------
    image_set : numpy array
        A dataset containing processed IR subframes.
        
    labels : numpy array
        Labels corresponding to the image set.
        
    num : integer
        The number of subframes in the completed test set.

    Returns
    -------
    test_image_set : numpy array
        Randomly generated test set.
        
    test_labels : numpy array
        Labels corresponding to the test set

    """
    
    test_image_set = []
    test_labels = []
    lst_index = []
    
    # Loop through num times
    for i in range (num):
        
        # Choose a random index that hasn't been chosen yet
        mask = ~np.isin(np.arange(labels.shape[0]), lst_index)
        choose = np.arange(labels.shape[0])[mask]
        index = np.random.choice(choose)
        
        # Append test data set and keep track of index used
        test_image_set.append(image_set[index])
        test_labels.append(labels[index])
        lst_index.append(index)
    
    test_image_set = np.array(test_image_set)
    test_labels = np.array(test_labels)
    
    return test_image_set, test_labels

class BlobDataset(Dataset):
    
    """Transform dataset into a pytorch compatible form.
    
    __init__ initializes the image set and labels.
    __len__ returns the length of the data set.
    __getitem__ returns the image and label for a given index.
    
    """
    
    def __init__(self, image_set, labels):
        
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        return image, label
    
class Flatten(nn.Module):
    
    """Flatten the last convolutional layer into an array using forward.
    
    """
    
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    
    """Unflatten the first decoder layer into convolutional blocks using forward.
    
    """
    
    def forward(self, input, shape_before_flatten):
        return input.view(input.size(0), *shape_before_flatten)
    
class Classifier(nn.Module):
    
    """Classify blob and non blob subframes using a CNN.
    
    """
    
    def __init__(self,
                 filters = [1, 8, 16],
                 neurons = [128, 2],
                 sub_array_size = 256,
                 k = 5,
                 pool = 2,
                 pad = 2,
                 dropout = 0.15):
        
        """Initial parameters and methods used to build the CNN.
        
        Parameters
        ----------
        filters : list
            A list of the number of filters at each convolutional layer.
            
        neurons : list
            A list of the number of neurons at each fully connected layer.
            
        sub_array_size : integer
            The length and width of the subframes.
            
        k : integer
            Kernel size for each convolutional layer.
            
        pool : integer
            Max pool size for each convolutional layer.
            
        pad : integer
            Zero padding on each side of the feature map.
            
        dropout : float
            Dropout regularization for fully connected layers.
            
        """
        
        super(Classifier, self).__init__()
        
        # Metadata
        self.sub_array_size = sub_array_size
        self.filters = filters
        self.pool = pool
        self.flat_neurons = filters[-1] * (sub_array_size // (pool ** (len(filters) - 1))) ** 2
        
        # Functions
        self.mp = nn.MaxPool2d(pool)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.logsm = nn.LogSoftmax(dim=1)
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=k, padding=pad)
        self.conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=k, padding=pad)
        
        # Dropout Rate
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(2 * dropout)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_neurons, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])


    def forward(self, x):
        
        """Push subframe through CNN for classification.
        
        The output is not the classification itself (0 or 1), 
        but the output neurons before classification (torch Tensor of size 2).
        The index of the output's neurons max value is the classification.
        We return the output instead of classification in order to utilize the
        Cross Entropy Loss function.
        
        Parameters
        ----------
        x : torch Tensor
            Input subframe.
                    
        Returns
        -------
        x : torch Tensor
            Output neurons. 
            
        """
        
        # Conv Layer 1 --> Activate --> Max Pool (256 --> 128)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Conv Layer 2 --> Activate --> Max Pool (128 --> 64)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Dropout 0.15 Neurons and Flatten (16 * 64 * 64 neurons)
        x = self.dropout1(x)
        x = self.flatten(x)
        
        #FC Layer 1 (Flatten --> 128)
        x = self.fc1(x)
        x = self.relu(x)
        
        #Dropuut 0.30 Neurons, FC Layer 2
        # (128 --> 2 classes)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
def count_parameters(model):
    
    """Count the number of trainable parameters in the CNN.
    
    Parameters
    ----------
    model : torch.nn.Module
        CNN model.
    
    Returns
    -------
    total_params
        Total trainable parameters in model.
    
    """
    
    total_params = 0
    
    for name, parameter in model.named_parameters():
        
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print([name, param])
        total_params += param
        
    return total_params

def format_dataset(image_set, labels, size=256):
    
    """Format the data set to feed into DataLoader.
    
    Parameters
    ----------
    image_set : numpy array
        A dataset containing processed IR subframes.
        
    labels : numpy array
        Labels corresponding to the image set.
        
    size : integer
        Length and width of the subframes.
        
    Returns
    -------
    data_set : list
        A dataset where an index returns the subframe and the label.
    
    """
    
    # Append subframes and corresponding labels to a list
    data_set = []
    for i in range(len(image_set)):
        data_set.append([image_set[i].reshape(1,size,size), labels[i]])
        
    return data_set

def train_model(train_loader, hyperparams):
    
    """Train CNN for a single epoch.
    
    Parameters
    ----------
    train_loader : torch.utils.data.dataloader.DataLoader
        Training set as a DataLoader.
        
    hyperparams : dictionary
        The model, loss function, optimizer, and device.
        
    Returns
    -------
    train_loss_norm : float
        Normalized training loss for one epoch.
        
    hyperparams_updated : dictionary
        Updated hyperparams after one epoch.
        
    """
    
    # Define hyperparameters
    model = hyperparams['model']
    distance = hyperparams['distance']
    optimizer = hyperparams['optimizer']
    device = hyperparams['device']
    
    # Change model to training mode (activates backpropogation)
    model.train()
    
    # Initialize training loss
    train_loss = 0
    
    # Loop through batches of training data
    for data, target in train_loader:
        
        # Put training batch on device
        data = data.float().to(device)
        target = target.type(torch.LongTensor).to(device)

        # Calculate output and loss from training batch
        output = model(data)
        loss = distance(output, target)
        
        # Backward Propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Normalize training loss from one epoch
    train_loss_norm = train_loss / len(train_loader)
    
    hyperparams_updated = {'model': model,
                           'distance': distance,
                           'optimizer': optimizer,
                           'device': device}
    
    return train_loss_norm, hyperparams_updated

def validate_model(valid_loader, hyperparams):
    
    """Validate CNN for a single epoch.
    
    Parameters
    ----------        
    valid_loader : torch.utils.data.dataloader.DataLoader
        Validation set as a DataLoader.
        
    hyperparams : dictionary
        The model, loss function, optimizer, and device.
        
    Returns
    -------
    val_loss_norm : float
        Normalized validation loss for one epoch.
        
    accuracy : float
        Model accuracy for one epoch.
        
    hyperparams_updated : dictionary
        Updated hyperparams after one epoch.
        
    """
    
    # Define hyperparameters
    model = hyperparams['model']
    distance = hyperparams['distance']
    optimizer = hyperparams['optimizer']
    device = hyperparams['device']
    
    # Change model to evaluate mode (deactivates backpropogation)
    model.eval()
    
    # Initialize validation loss and number of correct predictions
    val_loss = 0
    correct = 0
    
    # Do not backpropogate for evaluation loops
    with torch.no_grad():
        
        # Loop through batches of validation data
        for data, target in valid_loader:
            
            # Put validation batch on device
            data = data.float().to(device)
            target = target.type(torch.LongTensor).to(device)
            
            # Calculate output and loss from validation batch
            output = model(data)
            val_loss += distance(output, target).item()
            
            # Count number of correct predictions
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        
        # Calculate accuracy
        accuracy = 100. * correct / len(valid_loader.dataset)
    
    # Normalize validation loss from one epoch
    val_loss_norm = val_loss / len(valid_loader)
    
    hyperparams_updated = {'model': model,
                       'distance': distance,
                       'optimizer': optimizer,
                       'device': device}
    
    return val_loss_norm, accuracy, hyperparams

def build_model(train_image_set,
                train_labels,
                val_image_set,
                val_labels,
                dataloader_params,
                num_epochs,
                use_BlobDataset=False):
    
    """Train and validate CNN for a given number of epochs.
    
    Loss function is Cross Entropy Loss, which calculates classification 
    probabilities and determines the model's overall loss. Optimizer is Adam 
    (Adaptive Moment Estimation), which is an algorithm that optimizes the 
    learning rate. The device is cpu by default, but if gpu is availavle through 
    cuda, then the device uses cuda instead.
    
    Parameters
    ----------
    train_image_set : numpy array
        Training set of processed IR subframes.
        
    train_labels : numpy array
        Labels corresponding to the training set.
        
    val_image_set : numpy array
        Validation set of processed IR subframes.
        
    val_labels : numpy array
        Labels corresponding to the validation set.
        
    dataloader_params : dictionary
        Hyperparameters (batch_size, shuffle, num_workers) DataLoader uses
        to train the model.
    
    num_epochs : integer
        Number of epochs to train model.
        
    use_BlobDataset : boolean
        If True, uses BlobDataset class to format training and validation sets.
        Only make True if number of workers is greater than 2. If False, uses
        format_dataset for training and validation sets.
        
    Returns
    -------
    model_trained : nn.Module
        Trained CNN after a given number of epochs. 
        
    lst_train_loss : numpy array
        Training loss of the model at each epoch.
        
    lst_val_loss : numpy array
        Validation loss of the model at each epoch.
        
    lst_accuracy : numpy array
        Accuracy of the validation set at each epoch.
        
    """
    
    # Format training and validation set for DataLoader
    if use_BlobDataset:
        train_set = BlobDataset(train_image_set, train_labels)
        val_set = BlobDataset(val_image_set, val_labels)
    else:
        train_set = format_dataset(train_image_set, train_labels)
        val_set = format_dataset(val_image_set, train_labels)
    
    # Display number of times model will update
    num_model_updates = len(train_set) / dataloader_params['batch_size'] * num_epochs
    print ('Number of Model Updates: {}'.format(num_model_updates))
    
    # Load training and validation sets
    train_loader = DataLoader(train_set, **dataloader_params)
    valid_loader = DataLoader(val_set, **dataloader_params)
    
    # Prep model and device

    model = Classifier()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print (device)
    
    # Define hyperparameters
    hyperparams = {}
    hyperparams['distance'] = nn.CrossEntropyLoss()
    hyperparams['optimizer'] = torch.optim.Adam(model.parameters(), 
                                                weight_decay=1e-5)
    hyperparams['model'] = model
    hyperparams['device'] = device
    
    # Initialize metrics
    lst_train_loss = []
    lst_val_loss = []
    lst_accuracy = []
    
    # Loop through train model and validate model for a number of epochs
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        
        # Train model
        train_loss, hyperparams = train_model(train_loader, hyperparams)
        
        # Validate model
        val_loss, accuracy, hyperparams = validate_model(valid_loader, hyperparams)
        
        # Record metrics
        lst_train_loss.append(train_loss)
        lst_val_loss.append(val_loss)
        lst_accuracy.append(accuracy)

        # Log
        print('Epoch {:.3f} - Train loss: {:.3f} - Val Loss: {:.3f} - Accuracy: ({:.0f}%)'.format(
            epoch, train_loss, val_loss, accuracy))
    
    lst_train_loss = np.array(lst_train_loss)
    lst_val_loss = np.array(lst_val_loss)
    lst_accuracy = np.array(lst_accuracy)
    
    model_trained = hyperparams['model']
    
    return model_trained, lst_train_loss, lst_val_loss, lst_accuracy

def plot_metrics(num_epochs, lst_train_loss, lst_val_loss, lst_accuracy):
    
    """Plot training loss, validation loss, and accuracy as a function of epochs.
    
    Parameters
    ----------
    num_epochs : integer
        Number of epochs to train model.
        
    lst_train_loss : numpy array
        Training loss of the model at each epoch.
        
    lst_val_loss : numpy array
        Validation loss of the model at each epoch.
        
    lst_accuracy : numpy array
        Accuracy of the validation set at each epoch.
        
    Returns
    -------
    None
        
    """
    
    lst_epochs = np.arange(num_epochs)
    
    # Initialize subplots
    fig, axs = plt.subplots(1, 2, figsize=[20,10])

    # Plot loss functions
    axs[0].set_title('Train/Val Loss')
    axs[0].plot(lst_epochs, lst_train_loss, label='train')
    axs[0].plot(lst_epochs, lst_val_loss, label='val')
    axs[0].set_xlabel('Epochs')
    axs[0].legend()
    
    # Plot accuracy
    axs[1].set_title('Accuracy')
    axs[1].plot(lst_epochs, lst_accuracy)
    axs[1].set_xlabel('Epochs')

    plt.show()
    
    return None

def confusion_matrix(model, image_set, labels):
    
    """Calculate and plot confusion matrix of model given image set and labels.
    
    Parameters
    ----------
    model : nn.Module
        Trained CNN after a given number of epochs. 
        
    image_set : numpy array
        A dataset containing processed IR subframes.
        
    labels : numpy array
        Labels corresponding to the image set.
        
    Returns
    -------
    outputs : torch Tensor
        The output neurons of the CNN.
        
    predictions : numpy array
        Predictions made by the model corresponding to the image set.
        
    confusion_matrix : array like
        A confusion matrix of the model with predicted labels on the x axis and 
        true labels on the y axis.
    
    """
    
    # Change model to evaluate mode (deactivates backpropogation)
    model.eval()
    
    # Use trained model to make predictions from image set
    outputs = model(torch.Tensor(image_set.reshape(image_set.shape[0],1,256,256)))
    predictions = outputs.data.max(1, keepdim=True)[1].detach().numpy().flatten()
    predictions = np.array(predictions)
    
    # Calculate confusion matrix from labels and predictions
    confusion_matrix = metrics.confusion_matrix(labels, 
                                                predictions, 
                                                normalize='true')
    
    # Plot confusion matrix
    plt.figure(figsize=(16,10))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    return outputs, predictions, confusion_matrix

def saliency_map(model, image, label, index):
    
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
    axs[0].set_title('Index {}'.format(index))
    axs[0].imshow(X[0, 0].detach().numpy(), cmap='Greys')
    axs[1].set_title('Saliency Map for Index {}'.format(index))
    axs[1].imshow(sal_map, cmap=plt.cm.hot)
    
    if Y == float:
        print ('True Label: {}'.format(int(Y)))
    else:
        print ('True Label: {}'.format(Y))
        
    print ('Prediction: {}'.format(score_max_index))
    
    plt.show()
    
    return sal_map

def check_incorrect_image(image_set, labels, outputs, predictions, plot=True):
    
    """Check a random incorrect image and display model confidence.
    
    Parameters
    ----------
    image_set : numpy array
        A dataset containing processed IR subframes.
        
    labels : numpy array
        Labels corresponding to the image set.

    outputs : torch Tensor
        The output neurons of the CNN.
        
    predictions : numpy array
        Predictions made by the model corresponding to the image set.
        
    plot : boolean
        If True, plot a random incorrect image.
        If False, do not plot.
        
    Returns
    -------
    incorrect_image_set : numpy array
        Incorrectly classified image set.
        
    correct_labels : numpy array
        Correct labels corresponding to the incorrectly classified image set.

    incorrect_outputs : torch Tensor
        The output neurons of the incorrectly classified image set.
        
    predictions : numpy array
        Incorrect predictions made by the model corresponding 
        to the incorrectly classified image set.
        
    None :
        If there are no incorrect images.

    """

    outputs = outputs.detach().numpy()
    
    # Make a mask of incorrect predictions and apply mask to image set and outputs
    mask = ~(predictions == labels)
    incorrect_image_set = image_set[mask]
    incorrect_outputs = outputs[mask]
    
    # If there are incorrect images, choose a random one and plot
    if incorrect_image_set.shape[0] != 0:
        
        # Apply mask to predictions and labels
        incorrect_predictions = predictions[mask]
        correct_labels = labels[mask]
        
        # Plot random incorrect image
        if plot:
            
            # Choose a random index from incorrect image set
            index = np.random.randint(incorrect_image_set.shape[0])

            # Retrieve incorrect image and output
            image = incorrect_image_set[index]
            output = incorrect_outputs[index]

            # Softmax output to calculate probabilities
            prob_0, prob_1 = (np.exp(output) / 
                              np.exp(output).sum() * 100)

            # Retrieve incorrect prediction and correct label
            prediction = incorrect_predictions[index]
            label = correct_labels[index]
            
            # Find index in original set
            original_index = np.where(output == outputs)[0][0]

            plt.figure(figsize=[10,10])
            plt.title('Index: {}'.format(original_index))
            plt.imshow(image.reshape(256, 256), cmap='Greys')
            plt.colorbar()
            plt.show()

            # Print inaccuracy, prediction, true label, and model confidence
            inaccuracy = incorrect_image_set.shape[0] / image_set.shape[0] * 100
            print ('% of incorrect predictions: {:.2f}'.format(inaccuracy))
            print ('Pred: {}, True: {}'.format(prediction, int(label)))
            print ('Non Blob Prob: {:.2f}, Blob Prob: {:.2f}'.format(prob_0, prob_1))
        
        return (incorrect_image_set, 
                correct_labels, 
                incorrect_outputs, 
                incorrect_predictions)
    
    # If there aren't any incorrect images, return None
    else:
        print ('There were no incorrect images in the dataset')
        
        return None