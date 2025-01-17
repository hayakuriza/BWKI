from PIL import Image
import numpy as np
import os
from matplotlib import cm
import csv
import tensorflow as tf
import plaidml.keras
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

from keras.layers import BatchNormalization, Activation, Conv2D, Add, Dropout, Input, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt
from skimage.transform import resize

def padd(image):
    '''
    Reshapes the non-square image by pasting
    it to the centre of a black canvas of size
    n*n where n is the biggest dimension of
    the non-square image.
    '''
    old_size = image.size
    max_dimension, min_dimension = max(old_size), min(old_size)
    desired_size = (max_dimension, max_dimension)
    position = int(max_dimension/2) - int(min_dimension/2)
    blank_image = Image.new("RGB", desired_size, color='black')
    if image.height<image.width:
        blank_image.paste(image, (0, position))
    else:
        blank_image.paste(image, (position, 0))
    return blank_image


print("fetching images")
imgnum = 8189
images = np.empty(shape=(imgnum,256,256,3))
for i in range(1,imgnum):
    img = Image.open('../images/image_' + str(i).zfill(5) + '.jpg')
    img = padd(img).resize((256,256), Image.ANTIALIAS)
    images[i-1] = np.array(img)
    if(i % 100 == 0):
        print(i)

#im2 = Image.fromarray(np.uint8((images[47])))
#im2.show()

print("fetching labels")
labels_numeric = np.genfromtxt('../labels.csv', delimiter=',', dtype=int)
print(labels_numeric[0])


print("starting shuffle")
def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


train_in, train_target = randomize(images, labels_numeric)

#print(train_target)
#im2 = Image.fromarray(np.uint8((train_in[2])))
#im2.show()

print("start one-hot encoding")
labels = np.zeros((imgnum, 103))
labels[np.arange(8189), train_target] = 1
#print(labels[0])










#########RESNET-BLOCK#############
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    x = Dropout(DropoutRatioResBlock)(x)
    return x

DropoutRatio = 0.5 #dropoutRate
DropoutRatioResBlock = 0.2
start_neurons = 16

#128 -> 64
inputs = Input((256,256,3))
conv1 = Conv2D(start_neurons * 1, (7, 7), activation=None, padding="same")(inputs)
conv1 = residual_block(conv1,start_neurons * 1)
conv1 = residual_block(conv1,start_neurons * 1)
conv1 = residual_block(conv1,start_neurons * 1, True)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(DropoutRatio/2)(pool1)

# 64 -> 32
conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
conv2 = residual_block(conv2,start_neurons * 2)
conv2 = residual_block(conv2,start_neurons * 2)
conv2 = residual_block(conv2,start_neurons * 2, True)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(DropoutRatio)(pool2)

# 32 -> 16
conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
conv3 = residual_block(conv3,start_neurons * 4)
conv3 = residual_block(conv3,start_neurons * 4, True)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(DropoutRatio)(pool3)

# 16 -> 8
conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
conv4 = residual_block(conv4,start_neurons * 8)
conv4 = residual_block(conv4,start_neurons * 8, True)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(DropoutRatio)(pool4)

flat = Flatten()(pool4)
dense1 = Dense(1024)(flat)
dense1 = Dense(103)(dense1)
outputs = Activation('sigmoid')(dense1)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#opt = keras.optimizers.SGD(lr=0.01,momentum=0.7)
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

#nur für GPU-Nutzung, sonst auskommentieren!!!
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)



checkpointer = tf.keras.callbacks.ModelCheckpoint('chkpnt_best.h5', save_best_only=True, mode = 'max', monitor='val_categorical_accuracy', verbose=1)
checkpointer2 = tf.keras.callbacks.ModelCheckpoint('chkpnt_alw.h5', verbose=1)
lrred = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.25, patience=6, verbose=1, mode='max', min_delta=0.0001, min_lr=0.00001)
history = model.fit(train_in, labels,
                    validation_split=0.1,
                    batch_size=20,
                    epochs=60,
                    shuffle=True,
                    callbacks=[checkpointer,
                               checkpointer2,
                               lrred])


# Plot training & validation accuracy values
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoch')

# Plot training & validation loss values
f.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('categorical_crossentropy')
plt.xlabel('Epoch')
plt.show(block=True)