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
    Opens a flt.fits file and logarithmically processes the data according to 
    the following method:
    - opens the flt file, extracts SCI array, and creates an array of image 
    data.
    - Sets all values in the image array that are less than one to be equal to 
    1.
    - Scales the image data logarithmically.
    - Uses min/max scaling to scale the pixel values to be between 0 and 1. 
    - Resizes the image array to the dimensions (256, 256).

    Parameters:
    -----------
    fpath: string
    The string that identifies the directory that the image is located in.

    Returns: numpy array of floats
    --------
    log_data: array of floats
    Array of logarithmically scaled image data.

    rootname: string
    A String containing the rootname of the image that was processed.
    '''

     #Unpacking the image data: 
        # get the boolean value of the subarray
    subarray_value = fits.getval(fpath, 'subarray')
    
    # If the subarray is false then it is full chip, so we get the 
    # two data arrays, minus the 6 dead rows of pixels
    if subarray_value == False:
        data1 = fits.getdata(fpath,'sci', 1)[:-3]
        data2 = fits.getdata(fpath, 'sci', 2)[3:]
        
        image_data = np.vstack((data1, data2))
        
    else:
        
        image_data = fits.getdata(fpath, 'sci', 1)
    
        # Clip low values 
    image_copy = image_data.copy() # Copy image_data array

    pos_mask = image_copy < 1 # make positive mask
    image_copy[pos_mask] = 1 # Make all values positive
        
        # log scale data 
    log_data = np.log10(image_copy)
            
        # Resize log scaled image 
    log_data = resize(log_data, (256,256), order=3) 
    
        # Min/max scale image array:
    scaled_image = (log_data - log_data.min())/(log_data.max() 
                                                  - log_data.min())
    
    if orig_data == True:
        return(scaled_image, image_data)
    else:
        return(scaled_image)
    
    
def augment(image):
    '''
    Takes in an array of image data, and performs data augmentation. The
    image has a 50% chance of being vertically flipped, and then a 50%
    chance of being horizontally flipped. Finally, a degree is chosen at
    random from the range (0,360], the image is rotated to that angle, and then
    the image is cropped to [c:c1,c:c1] where:
    c = int(np.ceil((1/(np.sqrt(2)))*((np.sqrt(2)-1)/2)*256))
    c1 = 256 - c
    
    The final image array will have the size (180, 180)

    Parameters:
    -----------
    image: array of floats
    Image data that needs to be augmented.

    Returns:
    --------
    augmented_image: array of floats
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
    
    
        
    return(augmented_image)


def process_augment(fpath):
    '''
    Take in the path to an image file, process the image, augment the image 10 
    times, and then save the image to the specified directory.

    Parameters:
    -----------
    fpath: string
    String specifying the path to the image file to be processed.

    Returns:
    --------
    None

    '''

    r1 = os.path.basename(fpath)
    r2 = r1.split('_')
    root = r2[0]

    processed_image = log_image_process(fpath)
    
    aug_images = [] # list of augmented images

    for i in range(0,10):
        aug_images.append(augment(processed_image)) # augment image and add to 
                                                    # image list
    
    # save the processed and augmented images to the specified directory
    np.savez_compressed(f'{save_dir}/{root}_processed.npz', 
                        no_aug=processed_image, aug=aug_images)
    print('Processed rootname: ', root) # print the rootname that was processed

    return()



def main(file_name):
    '''
    Take in a path to a csv file, read the csv file in as a dataframe, and pool
    the processing, augmentation, and saving processes of the images within that
    file.

    Parameters:
    -----------
    file_name: string
    String specifying the path to the csv file with the locations of the images
    to be processed.

    Returns:
    --------
    None

    '''

    # Read in dataframe
    df = pd.read_csv(file_name)

    # make list of the image locations
    image_locations = []
    

    # get image locations
    for i in range(len(df)):
        img_info = df.iloc[i]
        root = img_info['rootname']
        fdir = img_info['dir']

        fpath = f'{fdir}/{root}_flt.fits'
        image_locations.append(fpath)

    total = len(image_locations)
    # pooling 
    pool = Pool(os.cpu_count())
    list(tqdm.tqdm(pool.imap(process_augment, image_locations), total=total))
    pool.close()
    pool.join()

    return()


if __name__ == "__main__":

    file_name = str(sys.argv[1])
    # file name is the full path to the csv file containing data
    
    fname = os.path.basename(file_name)

    # indicate the directory that the data should be saved to:
    save_dir = str(input("Please indicate the directory to save processed data \
                          to: "))
    # saving to the appropriate directory 
    main(file_name) # Run main function



