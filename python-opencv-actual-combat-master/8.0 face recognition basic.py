import cv2 as cv
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('image/elon mask.png')
imgelon = cv.cvtColor(imgelon , cv.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('image/elon test.png')
imgtest = cv.cvtColor(imgtest , cv.COLOR_BGR2RGB)

faceLoc =  face_recognition.face_locations(imgelon)[0]
encodeElon = face_recognition.face_encodings(imgelon)[0]
cv.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),1)

faceLoctest =  face_recognition.face_locations(imgtest)[0]
encodeElontest = face_recognition.face_encodings(imgtest)[0]
cv.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),1)

results = face_recognition.compare_faces([encodeElon],encodeElontest)
faceDis = face_recognition.face_distance([encodeElon],encodeElontest)
cv.putText(imgtest,f'{results}{round(faceDis[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv.imshow('elon mask',imgelon)
cv.imshow('elon test',imgtest)
cv.waitKey(0)