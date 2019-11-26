import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

cascadePath = os.path.join('..', 'haarcascade_frontalface_default.xml')
modelPath = os.path.join('..', 'model-gpu2.hdf5')

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cascadePath)
model = keras.models.load_model(modelPath)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


while True:
    _, frame = camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype(float) / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi).argmax()
        label = class_labels[preds]

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    cv2.imshow('Face', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()
