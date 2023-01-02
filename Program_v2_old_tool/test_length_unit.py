# coding: utf-8
import cv2
import numpy as np
import math
import sys




def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]), 2) + math.pow((Point0[1]-PointA[1]), 2)
    distance = math.sqrt(distance)
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
        times = times+1
    if times == 2:
        distance1 = getDist_P2P(points[0], points[1])
        distance2 = getDist_P2P(points[0], points[2])
        distance3 = getDist_P2P(points[3], points[4])
        length_in_mm_1 = (distance1 / distance3) * 500
        length_in_mm_2 = (distance2 / distance3) * 500
        error_in_percent_1 = abs(length_in_mm_1 - 15) / 15
        error_in_percent_2 = abs(length_in_mm_2 - 15) / 15
        error_res_1 = "error_1:{} ".format(error_in_percent_1)
        error_res_2 = "error_2:{} ".format(error_in_percent_2)
        file_handle.write(error_res_1)
        file_handle.write(error_res_2)
        file_handle.write("\n")
        sys.exit(0)


img = cv2.imread(r'D:\Courses\SJTU-IPP-Program2022\serverV2\after_perspective\AF_1.jpg')
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
file_handle = open('error_result.txt', mode='w')
times = 0
x, y = img.shape[0:2]
img = cv2.resize(img, (int(y / 5), int(x / 5)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
print(corners)
print("len:", len(corners))
print(corners[0])
# corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
points = [corners[0][0], corners[1][0], corners[11][0]]
for point in points:
    point[0] = point[0] / 4
    point[1] = point[1] / 4
x, y = img.shape[0:2]
img = cv2.resize(img, (int(y / 4), int(x / 4)))
# print img.shape
# cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.namedWindow("image")
# cv2.namedWindow("image", cv2.WINDOW_FREERATIO)   # 窗口大小自适应比例
cv2.resizeWindow("image", 1740, 1227)
# cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)   # 窗口大小保持比例
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=points)
cv2.imshow("image", img)
while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break
cv2.waitKey(0)
cv2.destroyAllWindow()



