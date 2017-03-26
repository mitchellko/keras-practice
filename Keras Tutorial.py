# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 00:36:30 2017
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
@author: Mitchell
"""
# Edit %USERPROFILE%/.keras/keras.json or set Environment Varuiable befoere import
# set KERAS_BACKEND=theano
# set KERAS_BACKEND=tensorflow
# before
import keras
print( keras.__version__)
print( keras.backend.backend())

import numpy as np
np.random.seed(123)  # for reproducibility
	
# Load pre-shuffled MNIST data into train and test sets
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print( x_train.shape)

from matplotlib import pyplot as plt
plt.imshow(x_train[0])

if ( keras.backend.image_data_format() == 'channel_first'):
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) 
print( x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print( y_train.shape)
print( y_train[:10])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print( y_train.shape)

# Define model architecture.
from keras.models import Sequential
model = Sequential()

from keras.layers import Convolution2D
if ( keras.backend.image_data_format() == 'channel_first'):
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
else:
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))

print( model.output_shape)

from keras.layers import MaxPooling2D, Dropout
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

from keras.layers import Flatten, Dense 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit( x_train, y_train, batch_size=32, epochs=1, verbose=1,
          validation_data=(x_test, y_test))