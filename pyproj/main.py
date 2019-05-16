from PIL import Image
import numpy as np
from matplotlib import cm
import csv
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from skimage.transform import resize

#print("fetching images")
#image_arr = np.empty(shape=())


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

#test_image = Image.open('../images/image_00632.jpg')
#new_image = reshape(test_image)
#new_image.show()
#new_image = new_image.resize((250, 250), Image.ANTIALIAS)
#new_image.show()

imgnum = 8189
images = np.empty(shape=(imgnum,256,256,3))
for i in range(1,imgnum):
    img = Image.open('../images/image_' + str(i).zfill(5) + '.jpg')
    img = padd(img).resize((256,256), Image.ANTIALIAS)
    #img.show()
    images[i-1] = np.array(img)
    if(i % 100 == 0):
        print(i)

im2 = Image.fromarray(np.uint8((images[4821])))
im2.show()
