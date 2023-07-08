# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from functools import partial


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython import display
import time


# %%


# %%
y_real=np.load('attributes5.npy')

print(y_real.shape)

images=np.load('images.npy')


print(np.max(images))
images=np.float32(images)

images=(images*2)/255-1
print(np.max(images))

print(y_real.shape)
print(images.shape)


# %%
tf.random.set_seed(0)
batch_size=64
train_batches = tf.data.Dataset.from_tensor_slices((images, y_real)).shuffle(10000).batch(batch_size)


# %%
for x,y in train_batches.take(1):
    print(x.shape)
    print(np.max(x))
    print(y.shape)
    

# %%


# %%
#This is for Conditional GAN
# Create the discriminator.
class Discriminator(Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',input_shape=(64, 64, 3),activation=partial(tf.nn.leaky_relu,alpha=0.3))
    self.bn1=layers.BatchNormalization()
    self.conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',activation=partial(tf.nn.leaky_relu,alpha=0.3))
    self.bn2=layers.BatchNormalization()
    self.conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same',activation=partial(tf.nn.leaky_relu,alpha=0.3))
   
    self.conv4 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same',activation=partial(tf.nn.leaky_relu,alpha=0.3))
    
    self.conv5 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same',activation=partial(tf.nn.leaky_relu,alpha=0.3))
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(1, activation='sigmoid')

  def call(self, x,y):
    x = self.conv1(x)
    x=self.bn1(x)
    y = y[:, None, None]
    y = tf.repeat(y, 64*64, axis=1)
    y = tf.reshape(y, (-1, 64, 64, 5))
    
    y = self.conv2(y)
    y=self.bn2(y)
    
    x = tf.concat([x,y],axis=-1)
    x = self.conv3(x)
    
    x = self.conv4(x)
    x = self.conv5(x)
     
    x = self.flatten(x)
    x = self.dense1(x)
    return x
  
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

# %%
generator=Generator()
discriminator=Discriminator()

# %%
input_shape = (batch_size,100)
x = tf.random.normal(input_shape)

y=y_real[0:batch_size]

out=generator(x,y)
print(out.shape)

plt.imshow(out[0,:,:,:])


xd=tf.random.normal((4,64,64,3))
yd=tf.random.normal((4,5))

out=discriminator(xd,yd)
print(out.shape)
print(out)


y = y_real[:4]
y[:, None, None].shape
yy = tf.repeat(y[:, None, None], 64*64, axis=1)
tf.reshape(yy, (-1, 64, 64, 5)).shape

a=images[8]
a.shape
#plt.imshow(a)


# %%
#Loss


# 1 means real, 0 means fake
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# %%
generator_optimizer = tf.keras.optimizers.Adam(5e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %%
#Setting up the training loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 30
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# %%

#Training function

@tf.function
def train_step(images,y):
    b_size=y.shape[0]
    noise = tf.random.normal([b_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise,y, training=True)

      real_output = discriminator(images,y, training=True)
      fake_output = discriminator(generated_images,y, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
    
    return gen_loss,disc_loss
    

# %%
def generate_and_save_images(model, epoch, test_input):

  y=y_real[0:num_examples_to_generate]
  predictions = model(test_input,y, training=False)
  predictions=(predictions*127.5+127.5)/255

  fig = plt.figure(figsize=(15,15),dpi=300)

  for i in range(predictions.shape[0]):
    plt.subplot(6, 5, i+1)
    plt.imshow(predictions[i, :, :, :])
    head=str(int(y[i][0]))+str(int(y[i][1]))+str(int(y[i][2]))+str(int(y[i][3]))+str(int(y[i][4]))
    plt.title(head)
    plt.axis('off')
  plt.savefig('image_epoch_{:04d}.png'.format(epoch))
  #plt.show()
  

# %%

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    gls=[]

    dls=[]

    for image_batch,y in dataset:
      gl,dl=train_step(image_batch,y)
      gls.append(gl.numpy())
      dls.append(dl.numpy())

    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    
    print('Gen Loss', np.mean(gls))
    print('Disc Loss', np.mean(dls))
    
    file1="model_weights"+str(epoch+1)
    file2="disc_weights"+str(epoch+1)

    generator.save_weights(file1)
    discriminator.save_weights(file2)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

# %%
#Black hair, Male, Oval Face, Smiling, Young
print(y_real[0:30])

# %%
train(train_batches, EPOCHS)





