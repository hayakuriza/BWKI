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
from keras.applications.vgg16 import VGG16
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
permutation1 = np.random.permutation(imgnum)
images = np.empty(shape=(imgnum,256,256,3))
for i in range(1,imgnum):
    img = Image.open('../images/image_' + str(i).zfill(5) + '.jpg')
    img = padd(img).resize((256,256), Image.ANTIALIAS)
    images[permutation1[i-1]] = np.array(img)
    if(i % 100 == 0):
        print(i)


images /= 255.0
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


#train_in, train_target = randomize(images, labels_numeric)

#print(labels_numeric)
#im2 = Image.fromarray(np.uint8((images[2])))
#im2.show()

print("start one-hot encoding")
labels = np.zeros((imgnum, 103))
labels[permutation1, labels_numeric] = 1
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
inputs = Input((128,128,3))
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
pool4 = Dropout(DropoutRatio/2)(pool4)

flat = Flatten()(pool4)
#dense1 = Dense(1024)(flat)
dense1 = Dense(103)(flat)
outputs = Activation('sigmoid')(dense1)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#opt = keras.optimizers.SGD(lr=0.01,momentum=0.7)
opt = keras.optimizers.Adam(lr=0.001)




model2 = Sequential()
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(128, 128, 3)))
model2.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dense(103, activation="softmax"))



base=VGG16(include_top=False, weights='imagenet',input_shape=(128,128,3), pooling='avg')

model3 = Sequential()
model3.add(base)
model3.add(Dense(1024, activation='relu'))
model3.add(Dense(103, activation='softmax'))

base.trainable=False



model3.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model3.summary()

#nur fuer GPU-Nutzung, sonst auskommentieren!!!
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)



checkpointer = keras.callbacks.ModelCheckpoint('chkpnt_best.h5', save_best_only=True, mode = 'max', monitor='val_categorical_accuracy', verbose=1)
checkpointer2 = keras.callbacks.ModelCheckpoint('chkpnt_alw.h5', verbose=1)
lrred = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.25, patience=6, verbose=1, mode='max', min_delta=0.0001, min_lr=0.00001)
history = model3.fit(images, labels,
                    validation_split=0.2,
                    batch_size=32,
                    epochs=10,
                    shuffle=True,
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
