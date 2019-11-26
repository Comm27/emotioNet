import tensorflow as tf
from tensorflow import keras


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
