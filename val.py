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

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
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

from FID_IS_function import *

from generator_class import *

# %%

y_real=np.load('attributes5.npy',mmap_mode='c')
images=np.load('images.npy',mmap_mode='c')

####Take subset of attributes and images for faster calculation of FID as it does not vary much with number of images
subset=10000
y_real=y_real[0:subset]


tf.random.set_seed(10)
num_to_generate = 1000 #Generate 1000 fake images
input_shape = (num_to_generate,100)
x = tf.random.normal(input_shape)

#Choose any of the y1, y2 or y3. All the generated images should have the same attributes either y1, y2 or y3
y1=np.array([1,1,0,0,1]) 
y2 = np.array([1,0,0,1,1]) 

y3=np.array([1,1,0,1,1])

y_select=y1 ##Change it according to the desired attributes

repetitions = num_to_generate
y = np.float32(np.tile(y_select, (repetitions, 1))) ###Make 1000 copies of it for 1000 images

####### IF you want to generate multiple images with attributes from the first 1000 data of attributes5.npy file then uncomment the following line

#y=y_real[0:num_to_generate] 





# %%

images=images[0:subset]
images=np.float32(images)
images=(images*2)/255-1 #Normalize between -1 and 1

# %%


tf.random.set_seed(10)
generator=Generator()
generator.load_weights("model_weights23")



def image_generator(model, epoch, test_input):
  
  predictions = model(test_input,y, training=False)
  predictions=(predictions*127.5+127.5)/255
  

  fig = plt.figure(figsize=(15,15),dpi=100)

  for i in range(num_to_generate):
    plt.subplot(10, 10, i+1)
    plt.imshow(predictions[i, :, :, :])
    head=str(int(y[i][0]))+str(int(y[i][1]))+str(int(y[i][2]))+str(int(y[i][3]))+str(int(y[i][4]))
    plt.title(head,fontsize=7)
    
    plt.axis('off')
 
  plt.show()
  return predictions

# %%
#Black hair, Male, Oval Face, Smiling, Young

#Generate and Visualize the fake images
pred=image_generator(generator,0,x)

# %%
#####CALCULATE FID AND IS



#####Calculate Inception score for fake images

images1=pred.numpy() #Generated Images
images255=images1*127.5+127.5 #MAKE 0 to 255

images255 = resize_images(images255, (299,299,3))
is_avg, is_std = calc_IS(images255)
print('\n\nIS Score: ',is_avg)



####Calculate FID Score
images1=pred.numpy() #Generated Images
images2 = images #Real Images

# declare inception model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


# resize images to fit in InceptionV3
images1 = resize_images(images1, (299,299,3))
images2 = resize_images(images2, (299,299,3))

images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# calculate fid
print('Takes some time (20-50 seconds) to calculate the FID')
fid = calc_fid_score(model, images1, images2) #Takes some time (20-50 seconds) to calculate the FID
print('FID score: %.3f' % fid)



