#!/usr/bin/env python3
# coding: utf-8

# In[1]:


# get_ipython().system(' export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1')


# In[2]:

import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


import tensorflow as tf
import os

import numpy as np

labels = ['normal', 'carrying','threat']


# In[ ]:


rooth_path = os.getcwd()


# In[10]:


# Manipulatable Variables
img_size = 224


""" Gets images from directory and returns them in a class labelled numpy array """
def get_data(data_dir, label):
  data = []
  # for label in labels:
    # path = os.path.join(data_dir, label)
  path = data_dir
  print("Path" + path)
  class_num = labels.index(label)
  for img_path in os.listdir(path):
    try:
      img_arr = cv2.imread(os.path.join(path, img_path))[...,::-1]  # convert BGR to RGB format
      resized_arr = cv2.resize(img_arr, (img_size, img_size))       # Reshaping images to preferred size
      data.append([resized_arr, class_num])
    except Exception as e:
      print(e)

  return np.array(data)


# In[11]:


# Load the data

normal_train  = os.path.join(root_path, 'normal/train')
normal_val    = os.path.join(root_path, 'normal/val')
normal_test   = os.path.join(root_path, 'normal/test')

carrying_train  = os.path.join(root_path, 'carrying/train')
carrying_val    = os.path.join(root_path, 'carrying/val')
carrying_test   = os.path.join(root_path, 'carrying/test')

threat_train  = os.path.join(root_path, 'threat/train')
threat_val    = os.path.join(root_path, 'threat/val')
threat_test   = os.path.join(root_path, 'threat/test')


# In[12]:


normal_train_df = get_data(normal_train, 'normal')


# In[13]:


normal_val_df = get_data(normal_val, 'normal')


# In[14]:


normal_test_df = get_data(normal_test, 'normal')


# In[15]:


carrying_train_df = get_data(carrying_train, 'carrying')


# In[16]:


carrying_val_df = get_data(carrying_val, 'carrying')


# In[17]:


carrying_test_df = get_data(carrying_test, 'carrying')


# In[18]:


threat_train_df = get_data(threat_train, 'threat')


# In[19]:


threat_val_df = get_data(threat_val, 'threat')


# In[20]:


threat_test_df = get_data(threat_test, 'threat')


# In[21]:


# print(len(normal_train_df))
# print(len(normal_val_df))
# print(len (normal_test_df))

# print(len(carrying_train_df))
# print(len(carrying_val_df))
# print(len(carrying_test_df))

# print(len(threat_train_df))
# print(len(threat_test_df))
# print(len(threat_test_df))

train_df = np.concatenate((normal_train_df, carrying_train_df, threat_train_df), axis=0)
val_df = np.concatenate((normal_val_df, carrying_val_df, threat_val_df), axis=0)
test_df = np.concatenate((normal_test_df, carrying_test_df, threat_test_df), axis=0)

print(len(train_df))
print(len(val_df))
print(len(test_df))

# print(train_df.shape)


# In[22]:


# Data Augmentation (To Explore whether this helps)


# In[23]:


plt.figure(figsize = (5,5))
plt.imshow(threat_train_df[5][0])
plt.title(labels[threat_train_df[5][1]])


# In[ ]:


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train_df:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val_df:
  x_val.append(feature)
  y_val.append(label)


# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

y_train = np.array(y_train)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range = 30,
    zoom_range = 0.2,
    width_shift_range =0.1,
    height_shift_range=0.1,
    horizontal_flip = True,
    vertical_flip=False)

datagen.fit(x_train)


# In[ ]:


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(3, activation="softmax")) # Final layer should output 3 classes hence 3 classes

model.summary()


# In[ ]:


opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[ ]:


history = model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val), callbacks=[callback])


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['Normal (Class 0)','Carrying (Class 1)', 'Threat (Class 2)']))


# In[ ]:


x_test = []
y_test = []

for feature, label in test_df:
  x_test.append(feature)
  y_test.append(label)

predictions_test = model.predict(x_test)
predictions_test = predictions_test.reshape(1,-1)[0]
print(classification_report(y_test, predictions_test,target_names = ['Normal (Class 0)','Carrying (Class 1)', 'Threat (Class 2)'] ))


# In[ ]:


model.save('/content/drive/MyDrive/CS4243 Project/model')

