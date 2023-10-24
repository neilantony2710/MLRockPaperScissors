import os
from keras.models import load_model
import cv2
import numpy as np

# Building data set
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)
counter = 0
stat = False
while True:
    ret, frame = camera.read()
    if counter == 200:
        break
    if ret == True:
        cv2.rectangle(frame, (0, 0), (640, 640), (0, 255, 0), 2)

        roi = frame[0:640, 0:640]

        roiGREY = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blah, roiBW = cv2.threshold(roiGREY, 15, 255, cv2.THRESH_BINARY_INV)

        ctrs, abc = cv2.findContours(
            roiBW.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangles = []

        for i in ctrs:
            rectangles.append(cv2.boundingRect(i))

        for z in rectangles:
            cv2.rectangle(frame, (z[0], z[1]), (z[0] + z[2], z[1] + z[3]), (255, 0, 0))
        cv2.imshow("video", frame)
        cv2.imshow("video2", roiBW)

    if cv2.waitKey(3) & 0xFF == ord("s"):
        stat = True
        print("starting")

    if cv2.waitKey(3) & 0xFF == ord("q"):
        break
    if stat and counter < 200:
        pathName = "paper/" + str(counter) + ".png"
        cv2.imwrite(pathName, roiBW)
        counter += 1
        print(roiBW.shape)
camera.release()
cv2.destroyAllWindows()
