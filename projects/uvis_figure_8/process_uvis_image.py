#!/usr/bin/env python

"""Process WFC3/UVIS data to be prepared for machine learning modeling.

We prepare the data through various steps in order to simplify the image as
much as possible. Since astronomical data ranges several orders of magnitude,
we log scale the image to reduce the overall pixel value range and clip data
to contain 99.9% of the original pixels below a max threshold. We scale the
image so all our images are normalized to a gaussian. We mean pool the image,
which can be as large as 4096x4096, to 256x256 to reduce image input size and
retain as much information and global structure as possible. Finally if any
images are not 256x256, they are reshaped accordingly.

Authors
-------
    Frederick Dauphin, 2021
Use
---
    This script is intended to run via command line as such:
        >>> python process_uvis_image.py <filename>
    where file is the path for a list of WFC3/UVIS image.

Notes
-----
    This script is the data processing pipeline for images used in
    DeepWFC3's Figure 8 Detection Project.
"""

import argparse
import os

import numpy as np
from ginga.util.zscale import zscale
from astropy.io import fits
from multiprocessing import Pool

import torch
import torch.nn as nn

def process_uvis_image(file):
    """Process a UVIS image by:

    1. log scaling the image
    2. cliping max values to contain 99.9% of original pixels
    3. normalize the image to mean of 0 and sigma of 1
    4. mean pooling the image
    5. zero-padding/cropping if necessary to be 256x256

    The 256x256 numpy array will be saved with a .npz extension in the current working directory with a key named 'image'.

    Parameters
    ----------
    file : str
        The path for the UVIS image.

    Returns
    -------
    None

    Examples
    -------
    >>> file = '/some/path/for/a/UVIS_image/rootname_flt.fits'
    >>> process_uvis_image(file)
    >>> # Load image
    >>> image = np.load('rootname_processed.npz')['image']
    """

    # Retrieve UVIS image
    print (file)
    if file[-10] == 'j':
        file = file.replace('j_flt.fits', 'q_flt.fits')

    # See if file exists, if not append to failed list
    try:
        data = fits.getdata(file, 'sci', 1)
    except FileNotFoundError:
        print ('File does not exist; exiting script.')
        return None

    # Determine if image uses both chips and remove chip gap
    try:
        data2 = fits.getdata(file, 'sci', 2)[3:]
        data = np.vstack((data[:-3], data2))
    except KeyError:
        pass

    # Log scale
    data[data<1]=1
    data = np.log10(data)

    # Clip min and max values
    vmax = 0
    per = 0
    while per <= 0.999:
        vmax += 0.05
        per = data[data<vmax].shape[0] / (data.shape[0]*data.shape[1])
    data[data>vmax]=vmax

    # Normalize image
    data = (data - data.mean()) / data.std()

    # Mean pool the image to be 256x256
    length = data.shape[1]
    a = nn.AvgPool2d(int(np.round(length/256)))
    pool = a(torch.Tensor(data.reshape(1,data.shape[0],data.shape[1])))
    pool = pool.detach().numpy()[0]

    # Zero-pad to 256x256 if image is too small
    condition1 = (pool.shape[0] < 256) | (pool.shape[1] < 256)
    if condition1:
        blank = np.zeros((256,256))

        y_length = pool.shape[0]
        x_length = pool.shape[1]
        y_pix = 128-(y_length//2)
        x_pix = 128-(x_length//2)
        try:
            blank[y_pix:y_pix+y_length, x_pix:x_pix+x_length] = pool
        except:
            print ("Can't properly reshape; exiting script.")
            return None
        pool = blank

    # Crop to 256x256 if image is too big
    condition2 = (pool.shape[0] > 256) | (pool.shape[1] > 256)
    if condition2:

        y_length = pool.shape[0]
        x_length = pool.shape[1]
        y_pix = y_length-256
        x_pix = x_length-256

        pool = pool[y_pix:y_pix+y_length, x_pix:x_pix+x_length]

    # Save and compress image as np.array
    rootname = os.path.basename(file)[:9]
    path = '{}_processed.npz'.format(rootname)
    np.savez_compressed(path, image = pool)

    return None

def run_pool(filename):
    """Process WFC3/UVIS images using multiprocessing.

    Parameters
    ----------
    filename : string
        Path to a .txt file containing .fits files to be processed.

    Returns
    -------
    None
    """

    image_names = np.loadtxt(filename, dtype=str)

    pool = Pool(32)
    pool.map(process_uvis_image, image_names)
    pool.close()
    pool.join()

    return None

def parse_args():
    """Parse the command line arguments.

    Returns
    -------
    args : obj
        An agparse object containing all of the added arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename',
        help='A list of the UVIS fits files.')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    run_pool(args.filename)
