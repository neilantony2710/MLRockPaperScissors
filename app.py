import os
import keras
from keras.models import load_model
import cv2
import time
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
# Loading model
model = load_model("./models/model.h5")
# Defining labels for model prediciton
categories = ['rock', 'paper', 'scissors']

camera = cv2.VideoCapture(0)

# Record Predictions into list to average
predList = []
for i in range(50):
    print("")
print("Welcome to my computer vision rock paper scissors game!")
print("type start to begin")
user_inp = str(input())
gameState = True if user_inp=="start" else False
if gameState:
    print("Position your hand to your right, the program will start in:")
    # Countdown
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)

while gameState:
    ret, frame = camera.read()

    if ret == True:
        cv2.rectangle(frame, (0, 0), (640, 640), (0, 255, 0), 2)

        roi = frame[0:640, 0:640]

        roiGREY = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        notNeeded, roiBW = cv2.threshold(roiGREY, 165, 255, cv2.THRESH_BINARY_INV)
        #roiBW = cv2.adaptiveThreshold(roiGREY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5)

        currentIMG = cv2.resize(roiBW, (50, 50))

        currentIMG = np.reshape(currentIMG, (1, 50, 50, 1))
        pred = model.predict(currentIMG, verbose=None)
        pred = np.argmax(pred[0])
        predList.append(pred)

        cv2.putText(frame, categories[pred], (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('video', frame)
        #cv2.imshow('video1', roiBW)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

    if len(predList) > 100:
        break
camera.release()
cv2.destroyAllWindows()

if gameState:
    computerMove = categories[random.randint(0,2)] 
    userMove = categories[round((sum(predList)/len(predList)))]
    print("The computer used: "+ computerMove)
    print("You used: " + userMove)
    
    if userMove == computerMove:
        print("Draw!")
    
    elif userMove == "rock":
        if computerMove == "scissors":
            print("You Win!")
        else:
            print("You Loose!")

    elif userMove == "paper":
        if computerMove == "rock":
            print("You Win!")
        else:
            print("You Loose!")

    else:
        if computerMove == "paper":
            print("You Win!")
        else:
            print("You Loose!")

print("Thank you for playing!")
