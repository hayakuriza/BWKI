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

from tensorflow.keras.models import load_model

IMGSIZE = 224
modelFile = '0.95n.h5'
imageFile = '../sunflower.jpg'



model = load_model(modelFile)
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


img = Image.open(imageFile)
img = padd(img).resize((IMGSIZE, IMGSIZE), Image.ANTIALIAS)
imgarr = np.array(img).reshape(1,224,224,3)
imgarr = imgarr / 255

out = model.predict(imgarr)
out = out.reshape(102)
#print(out)
print("predicted index: ", np.argmax(out))
#print(np.argmax(out))

with open("../class_labels_datagen.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)

print("predicted name: ", array[np.argmax(out)])

print("confidence: ", out[np.argmax(out)])