# PictoLabel

This project uses a convolutional neural network to identify images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


```
Keras with Tensorflow v2 Backend - https://keras.io/#installation
Python 3.6 - https://www.python.org/downloads/
```

### Configuration

Your desired training images should be catagorized, then organized into their respective folders. These folders should then all be put into the "Training" folder (Or as otherwise configured below).

For example:
```
/Training/Apples/image0xx.png
/Training/Oranges/image0xx.png
/Training/Pineapples/image0xx.png
```

The following settings can be set in the config.ini file as desired.

```
[Configuration]
Resolution: 100
Learning_Rate: 0.01
Epochs: 90
Batch_Size: 64
Training_Folder_Name: Training
```

## Running the tests

Todo

## Built With

Todo

## Acknowledgments

* Training Images (Fruits) used from www.kaggle.com/moltean/fruits/version/60
* Based off Adrian's tutorial at pyimagesearch.com
