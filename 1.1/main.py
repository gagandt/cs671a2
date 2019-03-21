import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion',one_hot=True)

"""
# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))
"""

#print(data.train.images[0])

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)






