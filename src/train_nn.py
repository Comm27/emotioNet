import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from utils import text_to_numpy_arr, get_data_from_df
from emotioNet import EmotioNet

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
dataset_path = os.path.join('..', 'dataset', 'fer2013', 'fer2013.csv')

data_frame = pd.read_csv(dataset_path)

tr = data_frame.Usage == 'Training'
tst = data_frame.Usage != 'Training'


(trainX, trainY), (testX, testY) = get_data_from_df(data_frame)
print('TrainX shape: ', trainX.shape)
print('TrainY shape: ', trainY.shape)
print('TestX shape: ', testX.shape)
print('TestY shape: ', testY.shape)


model = EmotioNet.build(48, 48, 1, len(class_labels))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

trainX = trainX / 255
testX = testX / 255

trainY_cat = keras.utils.to_categorical(trainY, len(class_labels))
testY_cat = keras.utils.to_categorical(testY, len(class_labels))
# model = keras.models.load_model('model-gpu.hdf5')
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(
    trainX, trainY_cat, batch_size=32, epochs=15, 
    verbose=1, validation_data=(testX, testY_cat))

model.save('model-gpu2.hdf5')

predictions = model.predict(testX)
pred = predictions.argmax(axis=1)
report = classification_report(testY, pred, labels=class_labels)
print(report)


# Plot history
plt.figure()
plt.plot(np.arange(0, 15), history.history['acc'], label='train_acc')
plt.plot(np.arange(0, 15), history.history['val_acc'], label='val_acc')
plt.plot(np.arange(0, 15), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epochs #')
plt.ylabel('Loss / Accuracy')
plt.show()
