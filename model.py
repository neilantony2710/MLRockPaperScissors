import os
import sklearn.model_selection
import keras
from keras.models import load_model
import cv2
import time
import numpy as np

categories = ["rock", "paper", "scissors"]
labels = []
data = []

for x in categories:
    y = os.listdir(x)
    for z in y:
        currentIMG = cv2.imread(os.path.join(x, z))
        currentIMG = cv2.cvtColor(currentIMG, cv2.COLOR_BGR2GRAY)

        waste, roiBW = cv2.threshold(currentIMG, 15, 255, cv2.THRESH_BINARY)

        currentIMG = cv2.resize(roiBW, (50, 50))
        data.append(currentIMG)
        labels.append(categories.index(x))


labels = np.array(labels)
data = np.array(data)
data = np.reshape(data, (600, 50, 50, 1))

(
    train_images,
    test_images,
    train_labels,
    test_labels,
) = sklearn.model_selection.train_test_split(data, labels, test_size=0.1)


# train_images = train_images/255
# print(train_images.shape)

""" Building A neural network model """
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            filters=50, kernel_size=(3, 3), input_shape=(50, 50, 1), activation="relu"
        ),
        keras.layers.MaxPooling2D((3, 3)),
        keras.layers.Conv2D(
            filters=50, kernel_size=(3, 3), input_shape=(50, 50, 1), activation="relu"
        ),
        keras.layers.MaxPooling2D((3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(3, activation="sigmoid"),
    ]
)
""" Compile the model """
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

""" Train the Model """
model.fit(train_images, train_labels, epochs=5)
# # """ Saving the Model"""
model.save("./models/model.h5")
