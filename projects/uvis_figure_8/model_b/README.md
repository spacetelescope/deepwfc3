## Model B
### Characteristics (from the ISR):
Model B, the best model, uses a 7-layer neural network architecture, with 3 convolutional layers and 4 fully connected layers. The sizes of the convolutional layers are 16, 32, 64 with a padding of 1 in each layer. Each convolutional layer is followed by a linear rectification (ReLU) and max pooling, with a  pooling size of 2. 
After passing through the convolutional layers, the output was flattened. 

The fully connected layers have 16, 32, 32 and 2 neurons, respectively. Each of the three first connected layers are followed by a ReLU and a dropout layer with a rate of 0.2 for regularization. 

### Load Model: 
To load the model we use the function `load_wfc3_fig8_modelb`. This function loads the model class (the architecture) of the model defined in utils (class Classifier). 
The trained model parameters are saved as `modelb_fig8_mixeddata.torch`. 

	model = load_wfc3_fig8_modelb('modelb_fig8_mixeddata.torch')
