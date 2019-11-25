import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
dataset_path = os.path.join('dataset', 'fer2013', 'fer2013.csv')

data_frame = pd.read_csv(dataset_path)

print(data_frame['pixels'])
print(data_frame.keys)