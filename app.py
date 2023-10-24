import os
import sklearn.model_selection
import keras
from keras.models import load_model
import cv2
import time
import numpy as np

# # Building data set
# camera = cv2.VideoCapture(0)
# camera.set(3, 640)
# camera.set(4, 480)
# counter = 0
# stat = False
# while True:
#     ret, frame = camera.read()
#     if counter == 200:
#         break
#     if ret == True:
#         cv2.rectangle(frame, (0, 0), (640, 360), (0, 255, 0), 2)

#         roi = frame[0:360, 0:640]

#         roiGREY = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#         blah, roiBW = cv2.threshold(roiGREY, 15, 255, cv2.THRESH_BINARY_INV)

#         ctrs, abc = cv2.findContours(
#             roiBW.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # cv2.imwrite(pathName, roiBW)
#         rectangles = []

#         for i in ctrs:
#             rectangles.append(cv2.boundingRect(i))

#         for z in rectangles:

#             cv2.rectangle(frame, (z[0], z[1]),
#                           (z[0]+z[2], z[1]+z[3]), (255, 0, 0))
#         cv2.imshow('video', frame)
#         cv2.imshow('video2', roiBW)

#         # counter += 1

#     if cv2.waitKey(3) & 0xFF == ord('s'):
#         stat = True
#         print("starting")

#     if cv2.waitKey(3) & 0xFF == ord('q'):

#         break
#     if stat and counter < 200:
#         # pathName = 'scissors/' + str(counter) + '.png'
#         # cv2.imwrite(pathName, roiBW)
#         counter += 1
#         print(roiBW.shape)
# camera.release()
# cv2.destroyAllWindows()

# HW: Mkae data set, build model and prediciton

categories = ['rock', 'paper', 'scissors']
# labels = []
# empty = []
# data = []

# for x in categories:

#     y = os.listdir(x)
#     for z in y:
#         currentIMG = cv2.imread(os.path.join(x, z))
#         currentIMG = cv2.cvtColor(currentIMG, cv2.COLOR_BGR2GRAY)

#         blah, roiBW = cv2.threshold(currentIMG, 15, 255, cv2.THRESH_BINARY)

#         currentIMG = cv2.resize(roiBW, (50, 50))
#         data.append(currentIMG)
#         labels.append(categories.index(x))


# labels = np.array(labels)
# data = np.array(data)


# train_images, test_images, train_labels, test_labels = sklearn.model_selection.train_test_split(
#     data, labels, test_size=0.1)

# train_images = train_images/255
# print(train_images.shape)

# ''' Building A neural network model '''
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(50, 50)),
#     keras.layers.Dense(200, activation='relu'),
#     keras.layers.Dense(3, activation='softmax')
# ])
# ''' Compile the model '''
# model.compile(optimizer='adam',
#               loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# ''' Train the Model '''
# model.fit(train_images, train_labels, epochs=3)
# # # """ Saving the Model"""
# model.save("rps.h5")
model = load_model("model.h5")
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
