# %%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from functools import partial
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython import display
import time

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize
from tensorflow.keras.datasets import cifar10
import time
import matplotlib.pyplot as plt
from numpy import asarray

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy import asarray


#Generator

class Generator(Model):
  def __init__(self):
    super().__init__()
    
    self.deconv1 = layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', activation='relu')
    self.bn1=layers.BatchNormalization()
    self.deconv2 = layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', activation='relu')
    self.bn2=layers.BatchNormalization()
    self.deconv3 = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', activation='relu')
    self.bn3=layers.BatchNormalization()
    self.deconv4 = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu')
    self.bn4=layers.BatchNormalization()
    self.deconv5 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')
    self.bn5=layers.BatchNormalization()
    self.deconv6 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')

    
  def call(self,x,y):
    x=x[:,None,None,:]
    x = self.deconv1(x)
    x=self.bn1(x)
    y = y[:, None, None]
    y=self.deconv2(y)
    y=self.bn2(y)
    x = tf.concat([x,y],axis=-1)
    x = self.deconv3(x)
    x=self.bn3(x)
    x = self.deconv4(x)
    x=self.bn4(x)
    x = self.deconv5(x)
    x=self.bn5(x)
    x = self.deconv6(x)
    return x