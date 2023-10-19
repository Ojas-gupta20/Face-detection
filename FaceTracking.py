import cv2
import numpy as np


w, h = 360, 240
fbRange=[6200,6800]
pid = [0.4, 0.4, 0]
pError = 0

def findFace(img):   # function to detect and track face and we have to find it in an image so we are passing that img
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")# here we are refering to haarcascade file for object detection using parameter and methods that is available on open cv
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.2,8)

    myFaceListC = []
    myFaceArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        cx = x + w//2
        cy = y + h//2
        area = w*h
        cv2.circle(img ,(cx,cy),5,(0,255,0),cv2.FILLED)
        myFaceListC.append(([cx,cy]))
        myFaceArea.append(area)

    if len(myFaceArea) != 0:
        i = myFaceArea.index(max(myFaceArea))
        return img, [myFaceListC[i], myFaceArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    area = info[1]

    x, y = info[0]

    fb = 0

    error = x - w // 2

    speed = pid[0] * error + pid[1] * (error - pError)

    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif  area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0

        error = 0

    print(speed, fb)

    

    return error





cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    img = cv2.resize(img, (w, h))
    img,info = findFace(img)
    pError = trackFace(info, w, pid, pError)
    print("Center", info[0], "Area", info[1])
    cv2.imshow("Output",img)
    cv2.waitKey(10)
