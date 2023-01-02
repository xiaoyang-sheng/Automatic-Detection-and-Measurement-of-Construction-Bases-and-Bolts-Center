# coding: utf-8
import cv2
import numpy as np
import math
import sys

img = cv2.imread(r'C:\Users\dell\Desktop\Courses\IPP\serverV2\calibresult3.jpg')
points=[]
times= 0
# print img.shape
def getDist_P2P(Point0,PointA):
    distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance=math.sqrt(distance)
    return distance

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%f,%f" % (x, y)
        print
        xy
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        points.append([x, y])
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        global times
        times=times+1
    if times==4:
        distance1 = getDist_P2P(points[0], points[1])
        distance2 = getDist_P2P(points[2], points[3])
        length_in_cm = distance1/distance2
        print("length of the width of blocks in cm is:")
        print(length_in_cm)
        sys.exit(0)

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)



while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break

cv2.waitKey(0)
cv2.destroyAllWindow()



