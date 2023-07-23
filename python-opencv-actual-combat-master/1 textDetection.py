import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt
tesseract_cmd = '"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"'

img = cv.imread("image/Text image.png")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)


## 框图框住单个字符
# print(pytesseract.image_to_string(img))
# hImg,wImg,_=img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#
#     print(b)
#     b = b.split(" ")
#     print(b)
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),3)
#     cv.putText(img,b[0],(x,hImg-y+25),cv.FONT_HERSHEY_COMPLEX,0.5,(50,50,255),2)

## 框图框住整体字符串
# data = pytesseract.image_to_data(img)
# for x,b in enumerate(data.splitlines()):
#     #x!=0，将第一个索引去掉，因为第一个索引是字符串
#     if x!=0:
#       b = b.split()
#       print(b)
#       if len(b) == 12:
#         x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#         cv.rectangle(img, (x,y), (w+x, h+y), (0, 0, 255), 3)
#         cv.putText(img, b[11], (w+x, h+y), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

##只检测单个数字
# hImg,wImg,_=img.shape
# cong = r"--oem 3 --psm 6 outputbase digits"
# boxes = pytesseract.image_to_boxes(img,config=cong)
# for b in boxes.splitlines():
#
#     print(b)
#     b = b.split(" ")
#     print(b)
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),3)
#     cv.putText(img,b[0],(x,hImg-y+25),cv.FONT_HERSHEY_COMPLEX,0.5,(50,50,255),2)

##检测数字组群
# cong = r"--oem 3 --psm 6 outputbase digits"
# data = pytesseract.image_to_data(img,config=cong)
# for x,b in enumerate(data.splitlines()):
#     #x!=0，将第一个索引去掉，因为第一个索引是字符串
#     if x!=0:
#       b = b.split()
#       print(b)
#       if len(b) == 12:
#         x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#         cv.rectangle(img, (x,y), (w+x, h+y), (0, 0, 255), 3)
#         cv.putText(img, b[11], (w+x, h+y), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)


plt.imshow(img)
plt.title("image")
plt.xticks([])
plt.yticks([])
plt.show()
