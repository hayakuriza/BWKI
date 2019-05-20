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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#VARS:
train_data_dir = '../categorys/'
input_size = 224
batch_size = 20
epochs = 10
drop_rate = 0.25

#DATAGEN
train_datagen = ImageDataGenerator(
    #preprocessing_function=keras.applications.resnet50.preprocess_input,
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #rotation_range=360,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #brightness_range=(0.8,1.2),
    #fill_mode='reflect',
    horizontal_flip=True,
    #vertical_flip=True,

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






DRM2 = 0.5

model2 = Sequential()
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(input_size, input_size, 3)))
model2.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(DRM2))

model2.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(DRM2))

model2.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(DRM2))

model2.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Flatten())
model2.add(Dropout(DRM2))
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dense(102, activation="softmax"))

#opt = keras.optimizers.Adam(lr=0.001)
#model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model2.summary()






#MODEL:
#base=VGG16(include_top=False, weights='imagenet',input_shape=(input_size,input_size,3), pooling='avg')
#base = ResNet50(include_top=False, weights='imagenet', input_shape=(input_size,input_size,3), pooling='avg')

base=MobileNet(include_top=False, weights='imagenet', input_shape=(input_size, input_size, 3), pooling='avg')
model = Sequential()
model.add(base)
model.add(Dropout(drop_rate))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(102, activation='softmax'))

base.trainable=False

#pt = keras.optimizers.RMSprop()
opt2 = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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

model.load_weights('chkpnt_best.h5')


for i in range (len(base.layers)):
    print (i,base.layers[i])



for layer in base.layers[82:]:
    layer.trainable=True
for layer in base.layers[0:82]:
    layer.trainable=False
opt2 = keras.optimizers.Adam(lr=0.001, decay=0.001)

model.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

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
g = plt.figure()
g.add_subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='acc')
plt.plot(history.history['val_categorical_accuracy'], label='val_acc')
plt.title('Model accuracy')
plt.legend(loc='upper left')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoch')

# Plot training & validation loss values
g.add_subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.legend(loc='upper left')
plt.ylabel('categorical_crossentropy')
plt.xlabel('Epoch')
plt.show(block=True)