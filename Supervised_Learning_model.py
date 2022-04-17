# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:13:50 2022

@author: Jiaxing Li
"""

import os
import cv2
import numpy as np 
import pandas as pd
import random as rn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

images = './Dataset/images'
annotations = './Dataset/annotations'

width = 80
height = 80
img_array = []
labels = []

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


classes = 3
encode = LabelEncoder().fit(labels)
encoded_y = encode.transform(labels)
y = np_utils.to_categorical(encoded_y, classes)
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42)

model = keras.models.Sequential([
    keras.layers.Conv2D(16, kernel_size=(5, 5), padding="same", strides=1, activation = 'relu', input_shape=(width, height, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", strides=1, activation = 'relu', input_shape=(width, height, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", strides=1, activation = 'relu', input_shape=(width, height, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])

b_size = 128
n_epochs = 32

opt = keras.optimizers.Adam()
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale = 1./255.,
                             rotation_range = 10,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             brightness_range = [0.2,1.2])

earlystopping = EarlyStopping(monitor ="val_loss", 
                                    mode ="auto", patience = 5, 
                                    restore_best_weights = True)

history = model.fit(datagen.flow(x_train, y_train, batch_size = b_size), 
                    epochs = n_epochs, 
                    validation_data=(x_val, y_val),
                    verbose=1, callbacks=[earlystopping])

model.evaluate(x_test, y_test)
model.save('model_classification.h5')
model.summary()
keras.utils.plot_model(model,show_shapes= True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()