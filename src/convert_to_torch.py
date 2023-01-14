from config import *

import torch
import keras
from torch import nn
from keras import layers

class PyTorchModel(nn.Module):
    def __init__(self, input_shape):
        super(PyTorchModel, self).__init__()
        self.zero_padding = nn.ZeroPad2d((2, 2))
        self.conv2d = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1))
        self.bn0 = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()
        self.max_pooling2d = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(6272, 1)
        
    def forward(self, x):
        x = self.zero_padding(x)
        x = self.conv2d(x)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.max_pooling2d(x)
        x = self.max_pooling2d_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Create a Pytorch model object
pytorch_model = PyTorchModel(input_shape=(240, 240, 3))

# Load the Keras model
keras_model = keras.models.load_model(KERAS_MODEL)

