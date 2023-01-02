# coding: utf-8
import cv2
import numpy as np
import math
import sys
import glob




def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]), 2) + math.pow((Point0[1]-PointA[1]),2)
    distance = math.sqrt(distance)
    return distance


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B


# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%f,%f" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         points.append([x, y])
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#         global times
#         times = times + 1
#     if times == 2:
#         distance1 = getDist_P2P(points[0], points[1])
#         distance2 = getDist_P2P(points[0], points[2])
#         distance3 = getDist_P2P(points[3], points[4])
#         length_in_mm_1 = (distance1 / distance3) * 500
#         length_in_mm_2 = (distance2 / distance3) * 500
#         error_in_percent_1 = abs(length_in_mm_1 - 15) / 15
#         error_in_percent_2 = abs(length_in_mm_2 - 15) / 15
#         error_res_1 = "error_1:{} ".format(error_in_percent_1)
#         error_res_2 = "error_2:{} ".format(error_in_percent_2)
#         print("error_res_1:", error_res_1)
#         print("error_res_2:", error_res_2)
#         file_handle.write(error_res_1)
#         file_handle.write(error_res_2)
#         file_handle.write("\n")
#         file_handle.flush()
#         print("one pic is done!")
#         times += 1
#         # sys.exit(0)
#         # return


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%f,%f" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        points.append([x, y])
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        global times
        times = times + 1
    if times == 4:
        len_side = getDist_P2P(points[0], points[1])
        len_1 = getDist_P2P(points[2], points[3])
        len_2 = getDist_P2P(points[3], points[4])
        len_3 = getDist_P2P(points[4], points[5])
        len_4 = getDist_P2P(points[5], points[2])
        length_in_mm_1 = (len_1 / len_side) * 15
        length_in_mm_2 = (len_2 / len_side) * 15
        length_in_mm_3 = (len_3 / len_side) * 15
        length_in_mm_4 = (len_4 / len_side) * 15
        error_len_1 = abs(length_in_mm_1 - 97.5) / 97.5 * 100
        error_len_2 = abs(length_in_mm_2 - 97.5) / 97.5 * 100
        error_len_3 = abs(length_in_mm_3 - 97.5) / 97.5 * 100
        error_len_4 = abs(length_in_mm_4 - 97.5) / 97.5 * 100
        angle_1 = cal_ang(points[2], points[3], points[4])
        angle_2 = cal_ang(points[3], points[4], points[5])
        angle_3 = cal_ang(points[4], points[5], points[2])
        angle_4 = cal_ang(points[5], points[2], points[3])
        error_ang_1 = abs(angle_1 - 90) / 90 * 100
        error_ang_2 = abs(angle_2 - 90) / 90 * 100
        error_ang_3 = abs(angle_3 - 90) / 90 * 100
        error_ang_4 = abs(angle_4 - 90) / 90 * 100
        print("length_in_mm_1: ", length_in_mm_1)
        print("length_in_mm_2: ", length_in_mm_2)
        print("length_in_mm_3: ", length_in_mm_3)
        print("length_in_mm_4: ", length_in_mm_4)
        print("error_len_1: ", error_len_1)
        print("error_len_2: ", error_len_2)
        print("error_len_3: ", error_len_3)
        print("error_len_4: ", error_len_4)
        print("angle_1: ", angle_1)
        print("angle_2: ", angle_2)
        print("angle_3: ", angle_3)
        print("angle_4: ", angle_4)
        print("error_ang_1: ", error_ang_1)
        print("error_ang_2: ", error_ang_2)
        print("error_ang_3: ", error_ang_3)
        print("error_ang_4: ", error_ang_4)
        string_1 = "length_in_mm: {}, {}, {}, {} ".format(length_in_mm_1, length_in_mm_2,
                                                          length_in_mm_3, length_in_mm_4)
        file_handle.write(string_1)
        file_handle.write("\n")
        string_2 = "error_len: {}, {}, {}, {} ".format(error_len_1, error_len_2, error_len_3, error_len_4)
        file_handle.write(string_2)
        file_handle.write("\n")
        string_3 = "angle_in_degree: {}, {}, {}, {} ".format(angle_1, angle_2, angle_3, angle_4)
        file_handle.write(string_3)
        file_handle.write("\n")
        string_4 = "error_ang: {}, {}, {}, {} ".format(error_ang_1, error_ang_2, error_ang_3, error_ang_4)
        file_handle.write(string_4)
        file_handle.write("\n")
        file_handle.write("\n")
        file_handle.flush()
        print("one pic is done!")
        times += 1
        # sys.exit(0)
        # return


images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\square_test_after_perspective\far\*.jpg')
file_handle = open('square_test_far_result.txt', mode='w')
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
# img = cv2.imread(r'D:\Courses\IPP\serverV2\c_3.jpg')
for fname in images:
    img = cv2.imread(fname)
    x, y = img.shape[0:2]
    times = 0
    img = cv2.resize(img, (int(y / 4), int(x / 4)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    points = [corners[0][0], corners[1][0]]
    # print(corners)
    # print(corners[0][0], corners[1][0], corners[2][0])
    # distance1 = getDist_P2P(points[0], points[1])
    # distance2 = getDist_P2P(points[0], points[2])
    # print("distance1:", distance1)
    # print("distance2:", distance2)
    for point in points:
        point[0] = point[0] / 3
        point[1] = point[1] / 3
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 3), int(x / 3)))
    cv2.namedWindow("image")
    # cv2.resizeWindow("image", 1740, 1227)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=points)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    continue
    # cv2.destroyWindow("image")
    # while (True):
    #     try:
    #         cv2.waitKey(100)
    #     except Exception:
    #         cv2.destroyWindow("image")
    #         break
    # cv2.waitKey(0)
    # cv2.destroyAllWindow()


# just tool function to get alphabet
def num_to_alphabet(num):
    aplha = "abcdefghijklmnopqrstuvwxyz"
    return aplha[num-1]




