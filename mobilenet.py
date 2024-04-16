import tensorflow as tf
import keras
import os
from keras.applications import MobileNetV3Small
from keras.layers import GlobalMaxPooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#load and prepare dataset
batch_size = 32
img_height = 450
img_width = 600
os.chdir("dataset")
data_dir = os.getcwd()

#load mobile net
mobilenet_pretrained= MobileNetV3Small(weights="imagenet", include_top = False, input_shape=(600,450,3))

#set all layers to not trainable
for layer in mobilenet_pretrained.layers:
    layer.trainable =  False

#add classification layers to the model
output = mobilenet_pretrained.layers[-1].output
x = GlobalMaxPooling2D()(output)
x = BatchNormalization()(x)
x = Dense(64, activation = "relu")(x)
classification = Dense(7, activation = "softmax")(x)
mobilenet = Model(mobilenet_pretrained.input, classification)
mobilenet.summary()
#

