import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('/home/szymon/Documents/my_whole_model')
model.summary()
