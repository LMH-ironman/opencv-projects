import cv2 as cv
import numpy as np

thres = 0.45
nms_threshold=  0.1
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


classnames =  []
classfile = "Object_Detection_Files/coco.names"
with open(classfile,"rt") as  f:
    classnames  = f.read().splitlines()

configPath = "Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath = "Object_Detection_Files/frozen_inference_graph.pb"

net = cv.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
   success,img = cap.read()
   classids , confs , bbox = net.detect(img,thres)
   bbox = list(bbox)
   confs = list(np.array(confs).reshape(1,-1)[0])
   confs = list(map(float,confs))
   indices = cv.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

   for i in indices:
      # i = i[0]
      box = bbox[i]
      x,y,w,h = box[0],box[1] ,box[2],box[3]
      cv.rectangle(img, (x,y),(x+w,h+y), (0, 255, 0), 5)
      cv.putText(img, classnames[classids[i]- 1].upper(), (box[0] + 10, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1,
                 (0, 0, 255), 2)
      cv.putText(img, str(round(confs[i] * 100, 2)), (box[0] + 200, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1,
                                  (0, 255, 255), 2)
   # if len(classids)  != 0:
   #    for classId,confidence,box in zip(classids.flatten(),confs.flatten(),bbox):
   #       print(classId,confidence,box)
   #       cv.rectangle(img,box,(0,255,0),5)
   #       cv.putText(img,classnames[classId-1].upper(),(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
   #       cv.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1,
   #                  (0, 255, 255), 2)
   cv.imshow("output",img)
   k = cv.waitKey(1)
   if k == 27:
      break
      cv.destroyAllWindows()
