# SISR implementation for aerial RGB imagery purposes

Single Image Super Resolution abbreviated as SISR is a neural network category which focuses on image reconstruction.

The code in this project is used to super resolve aerial imagery with the models [EDSR](https://arxiv.org/abs/1707.02921), [WDSR](https://arxiv.org/abs/1808.08718) and [SRGAN](https://arxiv.org/abs/1609.04802). This is a module of the project "Development of Data Fusion technique for mixed pixel decomposition and crop classification". The models trained in this module are especially trained for super resolving crop areas. You can retrain the models by preparing your own specific dataset as mentioned in the data preparation module.

For this project, programming environment mentioned in `environment.yml` is used. You need to prepare that programming environment before running any model mentioned in this directory.

Most of the implemented models are cloned from [this](https://github.com/krasserm/super-resolution) github repository and they are finetuned for our specific needs.

To train the models on your datasets you need to add a class that generates train and validation datasets to `data.py` and customize the code according to your need. You may need to change the variables that describe path in some files to avoid errors.

It is recommended to train the models for 300000 steps (which takes around 8 days for each model). Number of steps required can be customized according to the dataset.


## Directory Structure

### data
This directory contains dataset prepared from the drone images in `drone_rgb` direcotry

### drone_rgb
This directory contains two large rgb drone images (both of them contain crop areas the most)

### models
This directory contains necessary `.py` files necessary for the implementation of the models and some common utility functions that are used in the project.

### results
This directory contains results of super resolving a satellite images with both type of models:

1. Pretrained models on DIV2K dataset
2. trained models on custom drone image dataset saved in `data` directory

### train
This directory contains `.py` files used to train the models in differenct scenarios (local machine, gpus etc).

### weights
This directory contains the weights of the models that are trained on local machine for 3000 steps.

### weights_after_gpu_training
This directory contains model weights after the models are trained on gpu. Number of training steps are as follows:

1. EDSR - 30000 steps
2. WDSR - 30000 steps
3. EDSR + SRGAN - 15000 steps
4. WDSR + SRGAN - 15000 steps

### `custom_testing.py`
This file is used to test any extra functions or any tests that are needed to be conducted on the sisr project

### `data.py`
This file contains data loader class

### `train.py`
This directory contains model specific trainer functions

### `utils.py`
This directory contains common utility functions used across the whole project
