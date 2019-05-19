from PIL import Image
import numpy as np
from matplotlib import cm
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Add, Dropout, Input, MaxPooling2D, Flatten, Dense

from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#VARS:
train_data_dir = '../categorys/'
input_size = 128
batch_size = 20
epochs = 10

#DATAGEN
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.8,1.2),
    fill_mode='reflect',
    horizontal_flip=True,
    vertical_flip=True,

    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(input_size, input_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data



#MODEL:
base=VGG16(include_top=False, weights='imagenet',input_shape=(input_size,input_size,3), pooling='avg')

#base=Xception(include_top=False, weights='imagenet', input_shape=(IMGSIZE, IMGSIZE, 3), pooling='avg')
model = Sequential()
model.add(base)
model.add(Dense(1024, activation='relu'))
model.add(Dense(102, activation='softmax'))

base.trainable=False

opt = keras.optimizers.RMSprop()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()



checkpointer = tf.keras.callbacks.ModelCheckpoint('chkpnt_best.h5', save_best_only=True, mode = 'max', monitor='val_categorical_accuracy', verbose=1)
checkpointer2 = tf.keras.callbacks.ModelCheckpoint('chkpnt_alw.h5', verbose=1)
lrred = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.25, patience=6, verbose=1, mode='max', min_delta=0.0001, min_lr=0.00001)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = epochs,
    callbacks=[checkpointer,
               checkpointer2,
               lrred])


print("valid_classes")
print(validation_generator.classes)
print("valid_calss_indices")
print(validation_generator.class_indices)
print("train_classes")
print(train_generator.classes)
print("train_class_indices")
print(train_generator.class_indices)

# Plot training & validation accuracy values
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='acc')
plt.plot(history.history['val_categorical_accuracy'], label='val_acc')
plt.title('Model accuracy')
plt.legend(loc='upper left')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoch')


# Plot training & validation loss values
f.add_subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.legend(loc='upper left')
plt.ylabel('categorical_crossentropy')
plt.xlabel('Epoch')
plt.show(block=True)