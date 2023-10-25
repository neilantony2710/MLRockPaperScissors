import os
import sklearn.model_selection
import keras
from keras.models import load_model
import cv2
import time
import numpy as np

# Loading model
model = load_model("./models/model.h5")
# Defining labels for model prediciton
categories = ['rock', 'paper', 'scissors']

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if ret == True:
        cv2.rectangle(frame, (0, 0), (640, 640), (0, 255, 0), 2)

        roi = frame[0:640, 0:640]

        roiGREY = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        notNeeded, roiBW = cv2.threshold(roiGREY, 165, 255, cv2.THRESH_BINARY_INV)
        #roiBW = cv2.adaptiveThreshold(roiGREY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5)

        currentIMG = cv2.resize(roiBW, (50, 50))

        currentIMG = np.reshape(currentIMG, (1, 50, 50, 1))
        pred = model.predict(currentIMG)
        pred = np.argmax(pred[0])

        cv2.putText(frame, categories[pred], (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('video', frame)
        cv2.imshow('video1', roiBW)
    if cv2.waitKey(3) & 0xFF == ord('q'):

        break
camera.release()
cv2.destroyAllWindows()
