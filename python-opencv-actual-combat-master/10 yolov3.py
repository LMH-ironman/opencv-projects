import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
WHT =320
confThreshold = 0.9
nmsThreshold = 0.3

classfiles = 'Object_Detection_Files/coco.names'
classnames = []
with open(classfiles,'rt') as f:
    classnames = f.read().splitlines()


modelConfiguration = "D:\yolo\yolov3.cfg"
modelweights = "D:\yolo\yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelweights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT , wT ,cT = img.shape
    bbox = []
    classids = []
    confs = []

    for output in outputs:
        for det in output:

               scores = det[5:]
               classid = np.argmax(scores)
               confidence = scores[classid]
               if confidence > confThreshold:
                   w,h =  int(det[2]*wT) , int(det[3]*hT)
                   x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                   bbox.append([x,y,w,h])
                   classids.append(classid)
                   confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
         box  = bbox[i]
         x,y,w,h = box[0],box[1],box[2],box[3]
         cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
         cv.putText(img,f'{classnames[classids[i]].upper()} {int(confs[i]*100)}%',(x+200,y-10),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)




while True:
    success , img = cap.read()

    blob = cv.dnn.blobFromImage(img,1/255,(WHT,WHT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputnames = [layerNames[i-1]  for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputnames)
    findObjects(outputs,img)
    cv.imshow('img',img)
    k  = cv.waitKey(1)
    if k ==27:
        break
        cv.destroyAllWindows()