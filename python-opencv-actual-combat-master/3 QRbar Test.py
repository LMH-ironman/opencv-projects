import  cv2 as cv
import numpy as np
import pyzbar.pyzbar
from pyzbar.pyzbar import decode

# img = cv.imread("image/QR code.png")
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
# code = decode(img)
while True:
   success,img = cap.read()
   for barcode in  pyzbar.pyzbar.decode(img):
     # print(barcode.data)
     mydata = barcode.data.decode("utf-8")
     print(mydata)
     # print(success)
     # print(decode(img))
     pts = np.array([barcode.polygon],np.int32)
     pts = pts.reshape((-1,1,2))
     cv.polylines(img,[pts],True,(255,0,0),5)
     pts2 = barcode.rect
     cv.putText(img,mydata,(pts2[0],pts2[1]),cv.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
   cv.imshow("img",img)
   k = cv.waitKey(1)
   if k == 27:
     break
     cv2.destroyAllWindows()
