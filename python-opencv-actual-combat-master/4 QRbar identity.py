import  cv2 as cv
import numpy as np
import pyzbar.pyzbar
from pyzbar.pyzbar import decode

# img = cv.imread("image/QR code.png")
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
# code = decode(img)

with open("image/mydata.text") as f:
    mydatalist = f.read().splitlines()


while True:
   success,img = cap.read()
   for barcode in  pyzbar.pyzbar.decode(img):
     # print(barcode.data)
     mydata = barcode.data.decode("utf-8")
     print(mydata)
     # print(success)
     # print(decode(img))
     if mydata in mydatalist:
         Output =  "A"
         color = (0 , 255 , 0)
     else:
         Output =  "U-A"
         color = (0 , 0 , 255)
     pts = np.array([barcode.polygon],np.int32)
     pts = pts.reshape((-1,1,2))
     cv.polylines(img,[pts],True,(255,0,0),5)
     pts2 = barcode.rect
     cv.putText(img,Output,(pts2[0],pts2[1]),cv.FONT_HERSHEY_SIMPLEX,0.9, color,2)
   cv.imshow("img",img)
   k = cv.waitKey(1)
   if k == 27:
     break
     cv.destroyAllWindows()
