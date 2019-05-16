from PIL import Image
import numpy as np
from matplotlib import cm
import csv
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import optimizers
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
images = np.empty(shape=(imgnum,128,128,3))
for i in range(1,imgnum):
    img = Image.open('../images/image_' + str(i).zfill(5) + '.jpg')
    img = padd(img).resize((128,128), Image.ANTIALIAS)
    images[i-1] = np.array(img)
    if(i % 100 == 0):
        print(i)

#im2 = Image.fromarray(np.uint8((images[47])))
#im2.show()

print("fetching labels")
labels = np.genfromtxt('../labels.csv', delimiter=',')

print("starting shuffle")
def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


train_in, train_target = randomize(images, labels)

print(train_target)
im2 = Image.fromarray(np.uint8((train_in[2])))
im2.show()