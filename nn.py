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

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
dataset_path = os.path.join('dataset', 'fer2013', 'fer2013.csv')

data_frame = pd.read_csv(dataset_path)

tr = data_frame.Usage == 'Training'
tst = data_frame.Usage != 'Training'

def text_to_numpy_arr(txt):
    txt_arr = txt.split()
    return np.array(txt_arr).astype(int)

def get_data_from_df(df, shape=(48, 48, 1)):
    train = df.Usage == 'Training'
    test = df.Usage != 'Training'
    
    train_labels = df[train].emotion.to_numpy()
    test_labels = df[test].emotion.to_numpy()    
    
    tr = df[train].pixels
    tst = df[test].pixels
    
    train_imgs = [text_to_numpy_arr(x).reshape(shape) for x in tr]
    test_imgs = [text_to_numpy_arr(x).reshape(shape) for x in tst]
    
    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    return (train_imgs, train_labels), (test_imgs, test_labels)


(trainX, trainY), (testX, testY) = get_data_from_df(data_frame)
print('TrainX shape: ', trainX.shape)
print('TrainY shape: ', trainY.shape)
print('TestX shape: ', testX.shape)
print('TestY shape: ', testY.shape)


class EmotioNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=inputShape),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(classes, activation=tf.nn.softmax)
        ])
        
        return model


# model = EmotioNet.build(48, 48, 1, len(class_labels))
# model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

trainX = trainX / 255
testX = testX / 255

trainY_cat = keras.utils.to_categorical(trainY, len(class_labels))
testY_cat = keras.utils.to_categorical(testY, len(class_labels))
model = keras.models.load_model('model-gpu.hdf5')
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

