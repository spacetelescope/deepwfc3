WFC3 IR Blob Classification using Convolutional Neural Networks
---------------------

This repository contains the IPython Notebooks, data, and script needed to build the [WFC3 IR Blob Classifier](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2021/WFC3-2021-08.pdf). The purpose of this project is to automate the WFC3 Quicklook CSM monitor. The classifier will take subframes from a 1024x1024 IR blob difference image and predict if there is a prominent blob on the subframe. The Quicklook team will then be notified and record where new blobs appear over time.

Contents
----------------

**Notebooks:**

- `Data_Processing.ipynb`: The purpose of this notebook is to walk the user through the data processing pipeline for the WFC3/IR Blob Classifier. In this notebook, the user will process IR images and produce training, validation, and test sets.
- `Modeling_Evaluation.ipynb`: The purpose of this notebook is to walk the user through the modeling and evaluation pipeline for the WFC3/IR Blob Classifier. In this notebook, the user will train a convolutional neural network (CNN) to classify if blobs are in a subframe and evaluate the model's performance.

**Example Data (1024x1024 pixels):**

- `example_blob.fits`: IR blob image.
- `example_non_blob.fits`: IR blob difference image with no blobs.
- `example_median_stack`: IR blob median stack image.

**Scripts:**
- `wfc3_ir_blob_class.py`: This script contains all of our functions that will be used in the notebook. The two main functions are:
    - `make_dataset`, which pushes an IR image through our data processing pipeline and creates a data set by augmenting random subframes.
    - `build_model`, which trains and validates the model.

**Miscellaneous:**
- `blob_dict.npy`: A dictionary of 27 blobs that can be used for superimposing blobs onto non blob subframes.
