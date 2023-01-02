import cv2
import numpy as np
import glob
import math

def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]), 2) + math.pow((Point0[1]-PointA[1]),2)
    distance = math.sqrt(distance)
    return distance

def calculate_grid_length_stdev(corners):
    grid_length = []
    for i in range(0, 4):
        grid_length.append(getDist_P2P(corners[i][0], corners[i+1][0]))
    for i in range(15, 19):
        grid_length.append(getDist_P2P(corners[i][0], corners[i+1][0]))
    for i in range(0, 3):
        grid_length.append(getDist_P2P(corners[5*i][0], corners[5*(i+1)][0]))
        grid_length.append(getDist_P2P(corners[5*i+4][0], corners[5*(i+1)+4][0]))
    avg = np.average(grid_length)
    std = np.std(grid_length, ddof=1)
    # print("avg: ", avg)
    # print("std: ", std)
    # print("ratio: ", std/avg)
    return avg, std, std/avg

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\small_chessboard\l\*.jpg')
avgs = []
stds = []
ratios = []
for fname in images:
    img = cv2.imread(fname)
    x, y = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5, 4), None)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    avg, std, ratio = calculate_grid_length_stdev(corners)
    avgs.append(avg)
    stds.append(std)
    ratios.append(ratio)
# print(avgs)
# print(stds)
# print(ratios)
average_avg = np.average(avgs)
average_std = np.average(stds)
average_ratio = np.average(ratios)
print("average_avg: ", round(average_avg, 2))
print("average_std: ", round(average_std, 2))
print("average_ratio: ", round(average_ratio, 6))


