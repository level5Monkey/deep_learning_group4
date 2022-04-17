# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:12:44 2022

@author: Jiaxing Li
"""
import os
import cv2
import time
import PIL
import glob
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
import xml.etree.ElementTree as ET
from IPython import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# dataset
images = './Dataset/images'
annotations = './Dataset/annotations'
epochs = 500
width = 80
height = 80
image_size = 80
batch_size = 128
noise_dim = 100
num_examples_to_generate  = 16
img_array = []
labels = []
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def dim(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

for anot_file in sorted(os.listdir(annotations)):
    file_path = annotations + "/" + anot_file
    xml = ET.parse(file_path)
    root = xml.getroot()
    image = images + "/" + root[1].text

    # split pictures
    for bndbox in root.iter('bndbox'):
        [xmin, ymin, xmax, ymax] = dim(bndbox)
        img = cv2.imread(image)
        img_pr = img[ymin:ymax,xmin:xmax]
        img_pr  = cv2.resize(img_pr,(width, height))
        img_array.append(np.array(img_pr))
        
    # get labels
    for obj in root.findall('object'):
        name = obj.find('name').text 
        labels.append(np.array(name)) 
        
x = np.array(img_array)
x_train_grayscale = np.zeros(x.shape[:-1])
for i in range(x.shape[0]): 
    x_train_grayscale[i] = cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY)

x_train_grayscale = x_train_grayscale / 255.0
classes = 3
encode = LabelEncoder().fit(labels)
encoded_y = encode.transform(labels)
"""
0 mask_weared_incorrect
1 with mask
2 without mask
"""
x_train_grayscale.shape
encoded_y.shape

with_mask_dataset = x_train_grayscale[np.where(encoded_y==1)]
with_mask_dataset.shape
without_mask_dataset = x_train_grayscale[np.where(encoded_y==2)]
without_mask_dataset.shape
mask_weared_incorrect_dataset = x_train_grayscale[np.where(encoded_y==0)]
mask_weared_incorrect_dataset.shape

train_with_mask_dataset = tf.data.Dataset.from_tensor_slices(with_mask_dataset).shuffle(len(with_mask_dataset)).batch(batch_size=128)    
train_without_mask_dataset = tf.data.Dataset.from_tensor_slices(without_mask_dataset).shuffle(len(without_mask_dataset)).batch(batch_size=128)    

# plot
def plot_images(instances, images_per_row=4, **options):
    size = image_size
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = 'gray', **options)
    plt.axis("off")


plt.figure(figsize=(8, 8))
plt.subplot(111)

plot_images(with_mask_dataset[:4])
plot_images(without_mask_dataset[:4])
plot_images(mask_weared_incorrect_dataset[:4])

# model
generator_model = keras.models.Sequential([
    keras.layers.Input(100),
    keras.layers.Dense(20 * 20 * 256, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Reshape((20,20,256)),
    keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), use_bias=False, padding="same", strides=1),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), use_bias=False, padding="same", strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), use_bias=False, padding="same", strides=2, activation='tanh'),
    keras.layers.Reshape((image_size,image_size,1))])

generator_model.summary()

keras.utils.plot_model(generator_model,show_shapes= True)

discriminator_model = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", strides=(2, 2), input_shape=(image_size,image_size,1)),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", strides=(2, 2), input_shape=(image_size,image_size,1)),
    keras.layers.LeakyReLU(),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(1)])

discriminator_model.summary()

keras.utils.plot_model(discriminator_model,show_shapes= True)

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = keras.optimizers.Adam()
discriminator_optimizer = keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_model,
                                 discriminator=discriminator_model)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)
        
        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss
        print(disc_loss)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator_model,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator_model,
                           epochs,
                           seed)

train(train_with_mask_dataset, epochs)
 
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))  
display_image(epochs)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

# test the model using the images generated by the generator model  
model_classification = keras.models.load_model("model_classification.h5")
new_dataset = generator_model(tf.random.normal([100, 100]), training=False).numpy()
plot_images(new_dataset[:4])

test_dataset = np.arange(1920000)
test_dataset = np.reshape(test_dataset, (100,80,80,3))
test_dataset = np.zeros((100,80,80,3))

for i in range(len(new_dataset)):
    test_dataset[i] = cv2.cvtColor(new_dataset[i], cv2.COLOR_GRAY2RGB)

pred_y = model_classification.predict(test_dataset)
for data in pred_y:
    data[0] = 0 if data[0] < 0.5 else 1
    data[1] = 0 if data[1] < 0.5 else 1
    data[2] = 0 if data[2] < 0.5 else 1
true_y = np.full((100,3),[0,1,0])
accuracy_score(true_y, pred_y)


"""
res = generator_model(tf.random.normal([16, 100]), training=False).numpy()
res = res * 127.5
res = res + 127.5
plt.figure(figsize=(8, 8))
plt.subplot(111)
plot_images(res.reshape(16,80,80))    
"""