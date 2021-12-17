DeepWFC3: Analyzing HST/WFC3 Images using Machine Learning
--------

`deepwfc3` is a repository of machine learning models built using the Hubble Space Telescope's (HST) [Wide Field Camera 3 (WFC3)](https://www.stsci.edu/hst/instrumentation/wfc3) data. Under `projects`, the user will find folders for each model, which includes pretrained weights and biases, data processing scripts, Jupyter Notebooks, and a `README.md` briefly explaining the model's purpose. The models have an emphasis on anomaly detection in WFC3 images, which includes IR blobs, UVIS figure 8 ghosts, and more. For more information about WFC3 anomalies, please read [WFC3 ISR 2017-22](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-22.pdf).

Here is a list of the completed models:
- WFC3/IR Blob Classification using Convolutional Neural Networks (CNNs)
- WFC3/UVIS Figure 8 Classification using CNNs

In addition, we have some tutorials for implementing more advanced machine learning models in [PyTorch](https://pytorch.org/), our deep learning framework. Note the tutorials assume the user is familiar with machine learning basic vocabulary and methodology. They **DO NOT** act as a course for machine learning in general, but as a reference for complex models the developers had trouble learning about and implementing.

Here is a list of the available models with tutorials (all using [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data):
- Convolutional Neural Network
- Transfer Learning
- Autoencoders
- Variational Autoencoders

Installation
------------

All the libraries required for using the models are in `environment.yml`. The name of the anaconda virtual environment is `deepwfc3_env`, which contains standard scientific computing libraries (`numpy`, `matplotlib`, etc), machine learning frameworks (`pytorch` and `tensorflow`), and STScI libraries (`astropy`, `wfc3tools`, etc).

After cloning and changing directories to this repository, create the virtual environment by running this line in a terminal window:

```
conda env create -f environment.yml
```

To activate ``deepwfc3_env``, run this line in a terminal window:

```
conda activate wfc3_env
```

At the time this was written, the environment uses Python 3.6. It's within our best interest to use the latest software available so we will look into updating our environment sometime in the future.

Code of Conduct
------------

`deepwfc3` follows the [Astropy Code of Conduct](http://www.astropy.org/about.html#codeofconduct) and strives to provide a
welcoming community to all of our users and contributors.

License
-------

`deepwfc3` is licensed under a 3-clause BSD style license (see the ``LICENSE.txt`` file).
