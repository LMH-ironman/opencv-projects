import cv2 as cv
import numpy as np
import utlis

###########################################################

webCamFeed = True
pathimg = 'image/document scanner.png'
cap = cv.VideoCapture(0)
cap.set(10,160)
heightimg = 640
widthimg = 480



while True:
    if webCamFeed:success,img =cap.read()
    else:img = cv.imread(pathimg)
    img = cv.resize(img, (widthimg,heightimg))
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    cv.imshow('img',img)
    cv.waitKey(1)






