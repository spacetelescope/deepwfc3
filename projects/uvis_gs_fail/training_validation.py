# imports:
import time
import datetime
import glob
import sys
import os

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Import the model class:
from model import Model

# Define functions:
def get_data(path_list, label):
    '''Load in an array of images and create labels for each.

    Takes in a list of image paths, loads each image file, and creates arrays
    of non augmented and augmented data. Also creates arrays of labels for the 
    set of images.

    Parameters
    -----------
    path_list : list or array
        The paths to each image.
    label : string
        The type of data being processed. Allowed values are 'GS' and 'nominal'.
    
    Returns 
    --------
    non_aug_images : array
        The non-augmented images.
    y_non_aug : array
        The labels for the non-augmented images.
    augmented_images : array
        The augmented images.
    y_aug : array
        The labels for the augmented images.  
    '''
    # Initialize image lists
    non_aug_images = []
    aug_images = []
    
    for file in tqdm(path_list):
        full_info = np.load(file)
        non_aug = full_info['no_aug']
        non_aug_images.append(non_aug)
        aug = full_info['aug']

        # Bool to only load one aug image if label == nominal
        if label == 'GS':
            aug_images.append(aug)
        elif label == 'nominal':
            r = np.random.randint(0,9)
            aug_images.append(aug[r])
        
    # Combine the augmented image lists, and turn into arrays
    non_aug_images = np.array(non_aug_images)

    if label == 'GS':
        augmented_images = np.concatenate(aug_images)
    elif label == 'nominal':
        augmented_images = np.array(aug_images)
    
    # Create arrays of either 0's or 1's based on the label of the images
    if label == 'GS':
        y_non_aug = np.ones(len(non_aug_images))
        y_aug = np.ones(len(augmented_images))
    elif label == 'nominal':
        y_non_aug = np.zeros(len(non_aug_images))
        y_aug = np.zeros(len(augmented_images))
    
    print(f'finished loading data for: {os.path.dirname(path_list[0])}')

    # Return tuples of the arrays of image data
    return (non_aug_images, y_non_aug), (augmented_images, y_aug)

def organize_data(path, metric_dir):
    ''' Pull and label data to use in model training.

    Pulls random indices of training and validation data and organizes them into
    arrays of x (samples or features) and y (labels) data to prepare for model 
    training.

    Saves the indices pulled from the directories in a file in the directory 
    specified by 'metric_dir'. Assumes that 'path' has the subdirectories 
    training and test, which have 'GS' and 'nominal' subdirectories in them.

    Parameters
    -----------
    path : string
        The directory that the images are stored in.
    metric_dir : string
        The directory to save the indices of the images used in training and 
        validation sets and model evaluation metrics to.

    Returns
    --------
    augmented_data : list
        The augmented training and validation data sets with labels.
    non_augmented_data : list
        The non-augmented training and validation data sets with labels.
    '''
    T = datetime.datetime.now()
    dtime = T.strftime('%Y_%m_%d_%H-%M')

    # Get the length of the training directories:
    train_GS_paths = glob.glob(f'{path}/training/GS/*.npz')
    n_train = int((len(train_GS_paths))*10)

    test_GS_paths = glob.glob(f'{path}/test/GS/*.npz')
    n_test = int((len(test_GS_paths))*10)

    # Load the GS fail data:
    (trainGS_naug_x, trainGS_naug_y),(trainGS_aug_x, trainGS_aug_y) = \
                                                  get_data(train_GS_paths, 'GS')
    (testGS_naug_x, testGS_naug_y),(testGS_aug_x,testGS_aug_y) = \
                                                   get_data(test_GS_paths, 'GS')
    print('GS fail data loaded')
    
    # List of training paths
    paths_train = np.array(glob.glob(f'{path}/training/nominal/*.npz')) 
    # Array of indices the length of the path list
    train_ind = np.arange(len(paths_train), dtype=int) 
    # Shuffle the indices
    np.random.shuffle(train_ind)
    # Get indices to use (first n)
    train_ind_list = train_ind[:n_train] 
    # Get the paths from the path list
    train_data_short = paths_train[train_ind[:n_train]] 

    # Load the nominal data for the paths we just pulled:
    (trainnom_naug_x, trainnom_naug_y),(trainnom_aug_x, trainnom_aug_y) = \
                                           get_data(train_data_short, 'nominal')
    np.savetxt(f'{metric_dir}/train_indices_{dtime}.txt', train_ind_list) 
    print('Loaded shortened training data')

    # List of validation paths
    paths_test = np.array(glob.glob(f'{path}/test/nominal/*.npz'))
    # Create an array of indices
    test_indices = np.arange(len(paths_test), dtype=int) 
    # Shuffle the indices
    np.random.shuffle(test_indices)
    # Get indices to use (first n)
    test_ind_list = test_indices[:n_test]
    # Get the paths from the path list
    test_data_short = paths_test[test_indices[:n_test]]

    (testnom_naug_x, testnom_naug_y),(testnom_aug_x, testnom_aug_y) = \
                                            get_data(test_data_short, 'nominal')
    np.savetxt(f'{metric_dir}/test_indices_{dtime}.txt', test_ind_list) 
    print('Loaded shortened validation data')

    # Organize the augmented and non-augmented data
    # Training sets
    trainx_no_aug = np.concatenate((trainnom_naug_x,trainGS_naug_x))
    trainy_no_aug = np.concatenate((trainnom_naug_y,trainGS_naug_y))

    trainx_aug = np.concatenate((trainnom_aug_x,trainGS_aug_x))
    trainy_aug = np.concatenate((trainnom_aug_y,trainGS_aug_y))

    # Test sets:
    testx_no_aug = np.concatenate((testnom_naug_x,testGS_naug_x))
    testy_no_aug = np.concatenate((testnom_naug_y,testGS_naug_y))

    testx_aug = np.concatenate((testnom_aug_x,testGS_aug_x))
    testy_aug = np.concatenate((testnom_aug_y,testGS_aug_y))
    
    augmented_data = [trainx_aug, trainy_aug, testx_aug, testy_aug]
    non_augmented_data = \
                      [trainx_no_aug, trainy_no_aug, testx_no_aug, testy_no_aug]    
    
    print('finished loading and organizing data!')
    return augmented_data, non_augmented_data

def format_dataset(image_set, labels, size):  
    '''Reshape the dataset to prepare the data for model training.

    Reshapes the inpupt data to an array of size (1, image_size, image_size).
    Appends reshaped image array and the corresponding label to a list.

    Parameters
    -----------
    image_set : array
        The images that need to be formatted.
    labels : array
        The set of labels for the imaged being formatted.
    size : int or float
        The length of the images being formatted.

    Returns
    --------
    data_set : list
        The correctly formatted data set.
    '''
    data_set = []
    for i in range(len(image_set)):
        data_set.append([image_set[i].reshape(1,size,size), labels[i]])  
    return data_set

def save_model_metrics(metrics, model, data_type, metric_dir):
    ''' Saves the training and validation metrics for the model to a directory.

    Take in a list of model metric dataframes and save them to a specified
    directory. Also saves the final state dictionary for the model.

    Parameters
    -----------
    metrics : list
        The metrics to be saved.
    model : nn.Module
        The model that was trained.
    data_type : string
        The type of data the model will train on.
    metric_dir : string
        The directory to save the indices of the images used in training and 
        validation sets and model evaluation metrics to.

    Returns 
    --------
    metric_data : list
        The training and validation metric dataframes that were saved.
    '''
    val_metric_names = ['Epoch', 'Validation loss', 'Accuracy', 'Precision', \
                                                             'Recall', 'fscore']
    train_metric_names = ['Epoch', 'Epoch train time', 'Training loss', \
                                    'Accuracy', 'Precision', 'Recall', 'fscore']
    
    data = np.array(metrics[0])
    train_data = np.array(metrics[1])
    val_data = np.array(metrics[2])
    
    val_df = pd.DataFrame(np.transpose(val_data), columns=val_metric_names)
    train_df = pd.DataFrame(np.transpose(train_data), \
                                                     columns=train_metric_names)

    # save to csv file
    val_df.to_csv(f'{metric_dir}/val_model_{data_type}_{dtime}.csv') 
    train_df.to_csv(f'/{metric_dir}/train_model_{data_type}_{dtime}.csv') 
    
    # save model:
    torch.save(model.state_dict(), f'{metric_dir}/final_model_{data_type}_\
                                                                    {dtime}.pt') 
    metric_data = [train_df, val_df]

    return metric_data

class LoadDataset(Dataset):
    ''' Format the dataset so that the label and image are callable methods.

    Parameters
    ----------
    Dataset : array
        The dataset that needs to be reformatted.
    '''
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        return image, label
    
def train_model(train_loader):
    ''' Run the training procedure and return the model's metrics.
    
    Train the model and return the loss, accuracy, precision, recall, and 
    fscores for the particular epoch of model training.
    
    Parameters
    -----------
    train_loader: PyTorch tensor
        The images and labels.

    Returns
    --------
    train_loss_norm : float
        The normalized training loss for the current epoch.
    accuracy : float
        The accuracy of the model in the current epoch.
    precision : float
        The precision score of the model in the current epoch.
    recall : float
        The recall of the model in the current epoch.
    fscore : float
        The fscore of the model in the current epoch.
    '''

    # Change model to training mode (activates backpropogation)
    model.train()
    
    # Initialize training loss and number of correct predictions
    train_loss = 0
    correct = 0
    
    # Loop through batches of training data
    for data, target in train_loader:
        
        # Put training batch on device
        data = data.float().to(device)
        target = target.type(torch.LongTensor).to(device)

        # Calculate output and loss from training batch
        output = model(data)
        loss = distance(output, target)
        
        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Count number of correct preditions:
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        

    # Calculate accuracy
    accuracy = 100. * correct / len(train_loader.dataset)
    
    # Normalize training loss from one epoch
    train_loss_norm = train_loss / len(train_loader)

    # Calculate training precision and recall:
    precision,recall,fscore,support = precision_recall_fscore_support(target, \
                                       pred, average='micro', zero_division=1.0)
    
    return train_loss_norm, accuracy, precision, recall, fscore

def validate_model(valid_loader):
    '''Run the validation process and return the validation metrics for a model.

    Sets the model into eval mode, loop through the validation dataset from 
    valid_loader, and determine the loss, accuracy, precision, recall, and 
    fscore for the model on the dataset.

    Parameters
    ----------
    valid_loader : Torch tensor
        The images and labels.

    Returns 
    -------
    val_loss_norm : float
        The normalized validation loss metric for the current epoch.
    accuracy : float
        The accuracy of the model in the current epoch.
    precision :  float
        The precision metric of the model for the current epoch.
    recall :  float
        The recall metric for the current epoch.
    fscores : float
        The fscore for the current epoch.
    '''
    # Change model to evaluate mode (deactivates backpropogation)
    model.eval()
    
    # Initialize validation loss and number of correct predictions
    val_loss = 0
    correct = 0
    
    # Do not calculate gradients for the loop
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
        
        # Calculate validation and recall for the epoch
        precision,recall,fscores,support = precision_recall_fscore_support\
                             (target,pred, average = 'micro', zero_division=1.0)

    # Normalize validation loss from one epoch
    val_loss_norm = val_loss / len(valid_loader)
    
    return val_loss_norm, accuracy, precision, recall, fscores

def main(path, metric_dir, data_type):
    '''Load, organize, and format data sets, and train a model on that data.

    Performs the following steps for a model:
    1. Calls organize_data to get all the processed data and puts them into 
    tuples for augmented and non-augmented data.
    2. Uses a boolean to decide whether to use augmented or non-augmented data, 
    and splits the datasets into x and y datasets for training, and validation 
    sets.
    3. Set the hyperparameters, and format the training and validation sets as
    callable methods.
    4. Run the training and validation loops for the specified number of epochs
    and save the state dictionary every 10 epochs.
    5. Save the training and validation evaluation metrics for each epoch of 
    training to csv files.

    Parameters 
    -----------
    path : string
        The directory that the images are stored in.
    metric_dir : string
        The directory to save the model's evaluation metrics to.
    data_type : string
        The type of data that the model will train on. Allowed values are 
        'no_aug' and 'aug'.

    Returns
    --------
    metric_data : Dataframe
        The data for the loss, accuracy, precision, and recall metrics for both 
        the training and validation loops at each epoch of training.
    '''
    # Main directory to read files from
    augmented_data, non_augmented_data = organize_data(path)
    
    if data_type == 'aug':
        x_train = augmented_data[0]
        y_train = augmented_data[1]
        
        x_test = augmented_data[2]
        y_test = augmented_data[3]
        
    elif data_type == 'non_aug':
        x_train = non_augmented_data[0]
        y_train = non_augmented_data[1]
        
        x_test = non_augmented_data[2]
        y_test = non_augmented_data[3]

    # Dimensions for Model:
    x_train_size = x_train.shape[0]
    x_test_size = x_test.shape[0]
    x_length = x_train.shape[1]

    print('x_length = ', x_length)

    print('Image info being used in model:')
    print(x_train_size, x_test_size, x_length)
    
    # Format the dataset:
    train_set = format_dataset(x_train, y_train, x_length)
    val_set = format_dataset(x_test, y_test, x_length)
    
    ### SET HYPERPARAMETERS ###
    torch.manual_seed(42)

    params = {
            'batch_size': 128, 
            'shuffle': True,
            'num_workers': 0}

    # Define number of epochs:
    num_epochs = 100
    
    # Print the number of batches we will train for
    print ('The model will train using a total of {} batches'.format(num_epochs\
                                * int(x_train.shape[0] / params['batch_size'])))

    # TRAINING SET
    train_loader = DataLoader(train_set, **params)

    # TEST SET
    valid_loader = DataLoader(val_set, **params)

    ############ TRAINING AND VALIDATION ############
    # Track metrics:
    training_loss = []
    valid_loss = []
    # Accuracy
    val_accuracy = []
    train_accuracy = []
    # Recall
    val_recall = []
    train_recall = []
    # Precision
    val_precision = []
    train_precision = []
    # fscore
    val_f1 = []
    train_f1 = []
    # Epoch training time
    epoch_time = []

    # Training loop:
    for epoch in tqdm(range(num_epochs), total=num_epochs):

        # Start recording time
        t0 = time.time()
        # Go through loops
        train_loss, train_acc, train_prec, train_rec, train_fscore = \
                                                       train_model(train_loader)
        val_loss, accuracy, precision, recall, fscores = \
                                                    validate_model(valid_loader)
        # Stop recording time
        t1 = time.time()

        # Calculate time for the epoch
        dt = t1 - t0
        epoch_time.append(dt)

        # Append metrics
        # Loss
        training_loss.append(train_loss)
        valid_loss.append(val_loss)
        # Accuracy
        val_accuracy.append(accuracy)
        train_accuracy.append(train_acc)
        # Recall
        val_recall.append(recall)
        train_recall.append(train_rec)
        # Precision
        val_precision.append(precision)
        train_precision.append(train_prec)
        # fscore
        val_f1.append(fscores)
        train_f1.append(train_fscore)

        # Log training metrics
        print('Epoch {:.3f} - Train loss: {:.3f} - Val Loss: {:.3f} - Accuracy:\
                       ({:.0f}%)'.format(epoch, train_loss, val_loss, accuracy))
        
        # Every 10 epochs save model
        if (epoch % 10 == 0):
            # Save model params to specified directory:
            torch.save(model.state_dict(), f'{metric_dir}/model_epoch{epoch}_\
                                                        {data_type}_{dtime}.pt')

    # Array of the epoch numbers
    epoch = np.arange(num_epochs)

    # Evaluation metric lists:
    all_metrics = [(epoch+1), epoch_time, training_loss, train_accuracy, 
                   train_precision, train_recall, train_f1, valid_loss, 
                    val_accuracy, val_precision, val_recall, val_f1]
    val_metrics = [(epoch+1), valid_loss, val_accuracy, val_precision, 
                    val_recall, val_f1]
    train_metrics = [(epoch+1), epoch_time, training_loss, train_accuracy, 
                     train_precision, train_recall, train_f1]
    metrics = [all_metrics, train_metrics, val_metrics]

    # convert the metrics to dataframe; 
    metric_data = save_model_metrics(metrics, model, data_type, metric_dir)
    
    return metric_data

if __name__ == '__main__':
    
    # Directory to pull the files from:
    path = str(sys.argv[1])
    # Type of data that will be used (aug or non_aug)
    data_type = str(sys.argv[2])
    
    # Ask the user what 
    metric_dir = str(input('What directory would you like to save model \
                            metrics and state dictionaries to? '))

    if data_type == 'aug':
        sub_array_size = 180

    elif data_type == 'non_aug':
        sub_array_size = 256

    else:
        raise ValueError("Make sure to specify the type of data you want to \
                                                    use ('aug' or 'non_aug').")

    #initialize model:
    model = Model(sub_array_size=sub_array_size)

    # Loss function defining - Cross Entropy loss
    distance = nn.CrossEntropyLoss()

    # Optimizer definition: adam
    optimizer = torch.optim.Adam(model.parameters(),  weight_decay=1e-5)

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device);

    # list of labels:
    label_list = [0,1]

    # get the date/time string:
    t = datetime.datetime.now()
    dtime = t.strftime('%Y_%m_%d_%H-%M')

    # Call the main function to train the model:
    main(path, metric_dir, data_type)