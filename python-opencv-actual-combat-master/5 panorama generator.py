import cv2 as cv
import os

mainFolder = "image"
myFolders = os.listdir(mainFolder)
# print(myFolders)
img = []
# for folder in myFolders:
#     path = mainFolder + "/" + folder
#     # print(path)
#
#     # mylist = os.listdir(path)
#     # print(f"number {len(myFolders)}")

myimgN = myFolders[2:4]
print(myimgN)

for imgN in myimgN:
    newpath = mainFolder + "/" + imgN
    curimg = cv.imread(f"{newpath}")
    curimg = cv.resize(curimg,(0,0),None,0.2,0.2)
    img.append(curimg)


stitcher = cv.Stitcher.create()
(stastus,result) = stitcher.stitch(img)
if(stastus == cv.Stitcher_OK):
    print("Panorama created")
    cv.imshow(result)
    cv.waitKey(1)
else:
    print("not created")
cv.waitKey(0)



