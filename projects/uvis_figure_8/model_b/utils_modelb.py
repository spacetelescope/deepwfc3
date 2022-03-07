import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn


def process_image(image):
    X = torch.Tensor(image.reshape(1,1,256,256))
    return X
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Classifier(nn.Module):
    def __init__(self,
                 filters = [1, 16, 32, 64], # most people use powers of 2
                 neurons = [16, 32, 32, 2],  # neurons of fully connected layer
                 sub_array_size = 256,      # image size (256x256)
                 k = 3,                     # kernel size (3x3)
                 pool = 2,                  # pooling size (2x2)
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

        # ---- FULLY CONNECTED ----
        neurons_flat = filters[-1] * (sub_array_size // (pool**3))**2 #filters[-1] * (sub_array_size // pool ** 2) ** 2 # only works with k=3, pool=1
        self.fc1 = nn.Linear(neurons_flat, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])
        self.fc4 = nn.Linear(neurons[2], neurons[3])

        # ---- Batch Normalization
        self.bn3 = nn.BatchNorm2d(filters[2])

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.3)

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


        # Flatten Layer
        x = self.flatten(x)

        # Fully Connected 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fully Connected 2
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fully Connected 3
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fully Connected 4
        x = self.fc4(x)

        return x




def confusion_matrix(model, image_set, labels):
    from sklearn import metrics
    import seaborn as sns
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




def saliency_map(model, image):

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
    from matplotlib.colors import LogNorm
    # Rename image and label
    X = torch.Tensor(image.reshape(1,1,256,256))
    #Y = label
    # Change model to evaluation mode and activate gradient
    model.eval()
    X.requires_grad_()

    # Evaluate image and perform backwards propogation
    scores          = model(X)
    score_max_index = scores.argmax()
    score_max       = scores[0,score_max_index]
    score_max.backward()

    # Calculate saliency map
    saliency, _ = torch.max(X.grad.data.abs(),dim=1)
    sal_map     = saliency[0]

    softmax = torch.nn.Softmax(dim = 1)
    prob    = softmax(scores).detach().numpy().flatten()

    # if Y == float:
    #     print ('True Label: {}'.format(int(Y)))
    # else:
    #     print ('True Label: {}'.format(Y))

    print ('Predicted label: {}'.format(score_max_index))
    print ('Figure 8 probability: {}'.format(prob[1]))
    print ('')


    # Plot image and saliency map
    fig, axs = plt.subplots(1, 2, figsize=[16,8])
    axs[0].set_title('Input Image', fontsize = 20)
    axs[0].imshow(X[0, 0].detach().numpy(), cmap='Greys', origin = 'lower')
    axs[0].tick_params(axis = 'both', which = 'major', labelsize = 20)

    axs[1].set_title('Saliency Map', fontsize = 20)
    axs[1].imshow(sal_map, cmap=plt.cm.hot, origin = 'lower')
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.show()


    return sal_map


def load_wfc3_fig8_modelb(model_file):
    """
    Function to load the model B.
    First loads the model class (architecture). Then it loads the parameters of the trained model in model_file
    and changes it to evaluation mode.

    Parameters
    -------
    model_file : string
        File where model B is saved.

    Returns
    -------
    model :
        Model for classification of WFC3 Fig 8 ghost images.
    """
    model = Classifier()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    return model


# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
