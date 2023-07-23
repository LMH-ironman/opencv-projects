import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return  encodelist

def markAttendance(name):
    with open('imagesAttendance/attendance.csv', 'r+') as f:
        mydatelist = f.readline()
        namelist = []
        for line in mydatelist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodelistknown = findEncodings(images)



cap = cv.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceLoccapture = face_recognition.face_locations(imgS)
    encodecapture = face_recognition.face_encodings(imgS,faceLoccapture)

    for encodeface,faceLoc in zip(encodecapture,faceLoccapture):
        matchs = face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis = face_recognition.face_distance(encodelistknown,encodeface)
        matchsIndex = np.argmin(faceDis)

        if matchs[matchsIndex]:
            name = classNames[matchsIndex].upper()
            print(name)
            y1 , x2 ,y2 ,x1 = faceLoc
            y1, x2, y2, x1 =y1*4 , x2*4 ,y2*4 ,x1*4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv.imshow('webcap',img)
    k = cv.waitKey(1)
    if k == 27:
        break
        cv.destroyAllWindows()