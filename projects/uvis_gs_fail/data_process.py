# Imports:
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from astropy.io import fits
import tqdm
from skimage.transform import resize, rotate

def log_image_process(fpath, orig_data=False):
    '''
    Apply the logarithmic image processing pipeline to an image.

    Opens a flt.fits file and logarithmically processes the data according to 
    the following method:
        - Opens the SCI array of the flt file, and creates an array of image 
        data.
        - Sets all values in the image array that are less than one to be equal 
        to 1.
        - Scales the image data logarithmically.
        - Resizes the image array to the dimensions (256, 256).       
        - Uses min/max scaling to scale the pixel values to be between 0 and 1. 

    Parameters
    -----------
    fpath : string
        The path to the location of image to be processed. 
    
    orig_data : bool
        Specify whether or not to also return the original image data. Default
        value: False

    Returns
    --------
    scaled_image : array of floats
        The processed image.

    image_data : array of floats
        The original, pre-processed image data. Returned if orig_data==True
    '''
    # Get the boolean value of the subarray
    subarray_value = fits.getval(fpath, 'subarray')
    
    # If the subarray is false then it is full chip, so we get the two data
    # arrays, minus the 6 dead rows of pixels
    if subarray_value == False:
        data1 = fits.getdata(fpath,'sci', 1)[:-3]
        data2 = fits.getdata(fpath, 'sci', 2)[3:]
        
        image_data = np.vstack((data1, data2))
    else:
        image_data = fits.getdata(fpath, 'sci', 1)
    
    # Clip low values 
    # Copy image_data array
    image_copy = image_data.copy()

    # make positive mask
    pos_mask = image_copy < 1 

    # Make all values positive
    image_copy[pos_mask] = 1 
        
    # Log scale data 
    log_data = np.log10(image_copy)
            
    # Resize log scaled image 
    log_data = resize(log_data, (256,256), order=3) 
    
    # Min/max scale image array:
    scaled_image = (log_data - log_data.min())/\
                   (log_data.max() - log_data.min())
    
    if orig_data == True:
        return scaled_image, image_data
    else:
        return scaled_image
    
def augment(image):
    '''
    Performs data augmentation on image data.

    Take in an image and perform the following operations to create an augmented
    copy:
        - Vertically flip the image with 50% probability
        - Horizontally flip the image with 50% probability
        - Rotate the image to a random degree of (0,360]
        - Crop the image in the center to be (180,180)
    
    Parameters
    -----------
    image : array of floats
        Image data that needs to be augmented.

    Returns
    --------
    augmented_image : array of floats
        Augmented image data.
    '''
    # AUGMENT 1: VERTICAL FLIP
    flip_chance = np.random.random(2)
    
    # start with 50/50 chance of vertical flip
    if flip_chance[0] >= 0.5: 
        # flip image
        augment1 = np.flip(image, 0)
        
    elif flip_chance[0] < 0.5:
        # don't flip image
        augment1 = image
        
    # AUGMENT 2: HORIZONTAL FLIP
    # 50/50 chance of horizontal flip
    if flip_chance[1] >= 0.5:
        # flip image
        augment2 = np.flip(augment1, 1)
    elif flip_chance[1] < 0.5:
        # don't flip
        augment2 = augment1
        
    # AUGMENT 3: ROTATE AND CROP IMAGE
    # rotate image to some random degree in the range [0,360)
    R = np.random.random()
    rand_degree = (R*360) # degree value
    
    rotated_image = rotate(augment2, rand_degree) # rotate image
    
    # crop the image to the following coordinates:
    c = int(np.ceil((1/(np.sqrt(2)))*((np.sqrt(2)-1)/2)*256))
    c1 = 256 - c
    
    augmented_image = rotated_image[c:c1,c:c1] # cropped image

    return augmented_image

def process_augment(fpath):
    '''
    Take in the path to an image file, process the image, create 10 augmented 
    copies of the image, and then save the image and its copies to the specified
    directory.

    Parameters 
    -----------
    fpath : string
        The path to the image file to be processed.

    Returns
    --------
        none

    '''
    # Get the rootname of the image:
    r1 = os.path.basename(fpath)
    r2 = r1.split('_')
    root = r2[0]

    processed_image = log_image_process(fpath)
    # initialize empty list for augmented image
    aug_images = [] 

    for i in range(0,10):
        # augment image and add to image list
        aug_images.append(augment(processed_image)) 
    
    # save the processed and augmented images to the specified directory
    np.savez_compressed(f'{save_dir}/{root}_processed.npz', 
                        no_aug=processed_image, aug=aug_images)
    
    # print the rootname that was processed
    print('Processed rootname: ', root) 

def main(file_name):
    '''
    Take in a path to a csv file, read the csv file, and pool
    the processing, augmentation, and saving processes of the images within that
    file.

    Parameters
    -----------
    file_name : string
        The path to the csv file with the locations of the images to be 
        processed.

    Returns
    --------
        none
    '''
    # Read in csv file as dataframe
    df = pd.read_csv(file_name)

    # Make list of the image locations
    image_locations = []
    
    # Get image locations
    for i in range(len(df)):
        img_info = df.iloc[i]
        root = img_info['rootname']
        fdir = img_info['dir']

        fpath = f'{fdir}/{root}_flt.fits'
        image_locations.append(fpath)

    total = len(image_locations)
    # pooling image processing
    cpu_count = int(0.75*(os.cpu_count()))
    pool = Pool(cpu_count())
    list(tqdm.tqdm(pool.imap(process_augment, image_locations), total=total))
    pool.close()
    pool.join()

if __name__ == "__main__":

    file_name = str(sys.argv[1])
    # file_name is the full path to the csv file containing data
    
    fname = os.path.basename(file_name)

    # Indicate the directory that the data should be saved to:
    save_dir = str(input("Please indicate the directory to save processed data \
                          to: "))
    # Run main function
    main(file_name) 

