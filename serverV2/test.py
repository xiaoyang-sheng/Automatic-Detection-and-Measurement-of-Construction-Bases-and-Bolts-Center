import cv2
import numpy as np
import glob
import time

if __name__ == '__main__':
    images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2\after_perspective\AF_a.jpg')
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    for fname in images:
        img = cv2.imread(fname)
        x, y = img.shape[0:2]
        img = cv2.resize(img, (int(y / 5), int(x / 5)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        print("len: ", len(corners))
        print(corners)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (11, 8), corners, ret)
        cv2.imwrite("test_draw.jpg", img)
