import tensorflow as tf
from tensorflow.contrib import lite
print(tf.__version__)

converter = lite.
model = converter.convert()
file = open( 'model.tflite' , 'wb' )
file.write( model )