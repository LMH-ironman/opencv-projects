import  cv2 as cv
import math
import matplotlib.pyplot as plt

path = "image/Angle image.png"
img = cv.imread(path)
pointsList = []

def mousePoints(event,x,y,flags,params):
    if event == cv.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
             cv.line(img,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv.circle(img,(x,y),5,(0,0,255),cv.FILLED)
        pointsList.append([x,y])



def gradient(pt1,pt2):
    return (pt2[1] - pt1[1])/(pt2[0] - pt1[0])

def getAngle(pointsList):
    pt1, pt2 ,pt3 = pointsList[-3:]
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR =   math.atan((m2-m1)/(1+m1*m2))
    angD =  round(math.degrees(angR))  #round是四舍五入，math.degrees将弧度转换为度
    # print(angD)
    cv.putText(img,str(angD),(pt1[0]-40,pt1[1]-20),cv.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),2)

while True:

  if len(pointsList) % 3 == 0 and len(pointsList) != 0:
      getAngle(pointsList)

  cv.imshow("image",img)
  cv.setMouseCallback("image",mousePoints)
  if cv.waitKey(1) & 0xFF == ord("q"):
      pointsList = []
      img = cv.imread(path)

