import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
import pandas as pd
import math


# class to crop in the HED function below
class CropLayer(object):
    def __init__(self, params, blobs):
        # 初始化剪切区域开始和结束点的坐标
        self.xstart = 0
        self.ystart = 0
        self.xend = 0
        self.yend = 0

    # 计算输入图像的体积
    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # 计算开始和结束剪切坐标的值
        self.xstart = int((inputShape[3] - targetShape[3]) // 2)
        self.ystart = int((inputShape[2] - targetShape[2]) // 2)
        self.xend = self.xstart + W
        self.yend = self.ystart + H

        # 返回体积，接下来进行实际裁剪
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]




# used to calibrate camera
# param: image list, criteria
# output: camera matrix, new matrix, distortion coefficient
def get_camera_matrix(images, criteria):
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    file1 = images[0]
    img_1 = cv2.imread(file1)
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    size_all = gray_1.shape[::-1]
    cnt = 0
    for fname in images:
        cnt += 1
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                      cv2.CALIB_CB_FAST_CHECK +
                                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print("find chessboard corners!")
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            print("find corner_sub_pix!")
            img = cv2.drawChessboardCorners(gray, (11, 8), corners2, ret)
            # cv2.imshow('img', img)
    print(len(img_points))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size_all, None, None)
    # dist[0][4] = 0
    # if the calibration result is strongly twisted and has bubbles around, let k_3 = 0, but will reduce accuracy.
    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数
    print("-----------------------------------------------------")
    img = cv2.imread(images[1])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
    return newcameramtx, mtx, dist, roi


# function to do the perspective transform
# param: one pic, criteria, width, height, size, ratio
# output: the transformed image
def perspective_recover(dst, criteria, w, h, size=4000, ratio=0.03):
    # recover the perspective of the picture, i.e. make the chessborad in the z=0 surface.
    # - size: 导出图片的分辨率
    # - ratio：相当于棋盘格上一格占图片宽度的比例
    print("h:", h)
    print("w:", w)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    '''image = dst
    size = image.shape
    gray1 = cv.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv.COLOR_BGR2GRAY)'''
    # Find the chess board corners
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    '''for each in corners:
        if each is not None:
            each[0][1] += math.ceil(w / 3)
            each[0][0] += math.ceil(h / 4)'''
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    # uncomment this line to perform subpixel detection of corner, significantly slow the speed
    # prepare corresponding points for homography
    objp2 = np.zeros((11 * 8, 2), np.float32)
    cap = max(h, w)
    k = np.mgrid[-5:6:1, -4:4:1].T.reshape(-1, 2)
    objp2[:, :2] = (k * ratio + 0.5) * cap
    pts1 = np.float32([corners[0][0], corners[10][0], corners[-1][0], corners[-11][0]])
    # rectify picture to proper orientation that make the bolts face downward
    vec = corners[10][0] - corners[0][0]
    tan = vec[1] / vec[0]
    # !! only basic cases considered !!
    if tan <= 1 and tan >= -1:
        if vec[0] >= 0:  # 0 10 -1 -11
            pts2 = np.float32([objp2[0], objp2[10], objp2[-1], objp2[-11]])
        else:
            pts2 = np.float32([objp2[-1], objp2[-11], objp2[0], objp2[10]])
    else:
        if vec[1] >= 0:
            pts2 = np.float32([objp2[10], objp2[-1], objp2[-11], objp2[0]])
        else:
            pts2 = np.float32([objp2[-11], objp2[0], objp2[10], objp2[-1]])
    # obtain and perform transformation
    M = cv2.getPerspectiveTransform(pts1, pts2)
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    pts_ = cv2.perspectiveTransform(pts, M)
    [x_min, y_min] = np.int32(pts_.min(axis=0).ravel()-0.5)
    [x_max, y_max] = np.int32(pts_.max(axis=0).ravel()+0.5)
    diff = [-x_min, -y_min]
    H_diff = np.array([[1, 0, diff[0]], [0, 1, diff[1]], [0, 0, 1]])
    H = H_diff.dot(M)
    dst = cv2.warpPerspective(dst, H, (x_max-x_min, y_max-y_min))
    return dst


# tool function to get alphabet
def num_to_alphabet(num):
    aplha = "abcdefghijklmnopqrstuvwxyz"
    return aplha[num-1]


# function that call the get_camera_matrix and perspective_recover, includes some file IO
def calibration_and_perspective():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\camera_undistort_8\*.jpg')
    images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\test_pic_2\*.jpg')
    print("load the pictures!")
    newcameramtx, mtx, dist, roi = get_camera_matrix(images_cam, criteria)
    cnt = 0
    for fname in images_cam:
        print(fname)
        cnt += 1
        img = cv2.imread(fname)
        h_, w_ = img.shape[:2]
        dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst2 = perspective_recover(dst1, criteria, w_, h_)
        file_name = "./tt/AF_" + num_to_alphabet(cnt) + ".jpg"
        cv2.imwrite(file_name, dst2)
        print("已写入:", file_name)


# function to get HED edge detection
def HED(image, net, downW=800, downH=800):
    (H, W) = image.shape[:2]
    image = cv2.resize(image, (downW, downH))
    # 根据输入图像为全面的嵌套边缘检测器（Holistically-Nested Edge Detector）构建一个输出blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(800, 800),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=True)
    # # 设置blob作为网络的输入并执行算法以计算边缘图
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    # 调整输出为原始图像尺寸的大小
    hed = cv2.resize(hed[0, 0], (W, H))
    # 将图像像素缩回到范围[0,255]并确保类型为“UINT8”
    hed = (255 * hed).astype("uint8")
    return hed


# function to detect the outside base circle with HoughCircle and HED
# return the circle center
def detect_direct_hough_with_hed(image, net):
    hed = HED(image, net)
    (H, W) = hed.shape[:2]
    circleshed = cv2.HoughCircles(hed, cv2.HOUGH_GRADIENT, 1, round(H / 2),
                                 param1=100, param2=60,
                                 minRadius=round(H / 6), maxRadius=round(H / 2))
    center_x = 0
    center_y = 0
    for i in circleshed[0, :]:  # 遍历矩阵每一行的数据
        cv2.circle(image, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        cv2.circle(image, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
        # above is to draw the circle and circle center on image, if you would like to check, you could use imshow
        center_x = i[0]
        center_y = i[1]
    return center_x, center_y


# function to call detect_direct_hough_with_hed
def detect_base():
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # above is the preparation for usage of HED, these two files are downloaded before
    cv2.dnn_registerLayer("Crop", CropLayer)
    images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model_perspective\*.jpg')
    fname = images_cam[0]
    img = cv2.imread(fname)
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 24), int(x / 24)))
    x_center, y_center = detect_direct_hough_with_hed(img, net)
    print("x_center: ", x_center)
    print("y_center: ", y_center)


# function to detect the four bolts bottom and calculate the center of those four circle centers
def local_grid(gray, param):
    # detect the center of bolts relative to the centre of
    (H, W) = gray.shape[:2]
    circleshed = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=100, param2=param,
                                 minRadius=30, maxRadius=40)
    # use K-means algorithm to group the circles and extract the minimum from each groups
    km = KMeans(n_clusters=4).fit(circleshed[0, :, 0:2])
    df = pd.DataFrame({'X': circleshed[0, :, 0],
                       'Y': circleshed[0, :, 1],
                       'Radius': circleshed[0, :, 2],
                       'Label': km.labels_}
                      )
    mins = df.sort_values('Y', ascending=False).groupby('Label', as_index=False).first()
    # show the circles
    for index, i in mins.iterrows():
        center = (i['X'].astype("int"), i['Y'].astype("int"))
        cv2.circle(gray, center, 1, (255, 255, 255), 3)
        radius = i['Radius'].astype("int")
        cv2.circle(gray, center, radius, (255, 255, 255), 3)
    cv2.imshow("canny", gray)
    cv2.imwrite("af7_far_fan.jpg", gray)
    cv2.waitKey()
    return (mins['X'], mins['Y'])


# function to call local_grids
def detect_bolts():
    protoPath = "/Users/zhangzeyu/Desktop/college/IPP/server/serverV2/deploy.prototxt"
    modelPath = "/Users/zhangzeyu/Desktop/college/IPP/server/serverV2/hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cv2.dnn_registerLayer("Crop", CropLayer)
    img = cv2.imread("/Users/zhangzeyu/Desktop/college/IPP/close_photo/after_close/AF_14.jpg")
    size = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 按真实长度缩放
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    pixel = corners[12][0] - corners[13][0]
    d = math.sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1])
    ratio = d / 15
    ratiox = int(size[1] / ratio)
    ratioy = int(size[0] / ratio)
    img = cv2.resize(img, (ratiox, ratioy))
    # 裁剪
    size = img.shape
    img1 = img[math.ceil(size[0] / 10):math.ceil(9 * size[0] / 10),
           math.ceil(size[1] / 100):math.ceil(99 * size[1] / 100)]
    s = img1.shape
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = HED(img1, net)
    # 求螺栓中心坐标
    x, y = local_grid(gray2, 18)
    print(x, y)


# circle detection for bolts too, written by another student
def local_grid_2(gray, net):
    x_sum = 0
    y_sum = 0
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                  cv2.CALIB_CB_FAST_CHECK +
                                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
    center = corners[38]
    # actual center of four cylinders
    print('actual center: ', center)
    center_x = center[0][0]
    center_y = center[0][1]
    hed = HED(gray, net)
    x, y = hed.shape[0:2]
    cv2.imshow("hed", hed)
    cv2.waitKey(0)
    # detect the center of bolts relative to the centre of
    (H, W) = gray.shape[:2]
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny = cv2.Canny(blurred, 100, 100)
    # obtain edge through Canny alogrithm

    circleshed = cv2.HoughCircles(hed, cv2.HOUGH_GRADIENT, 1, 30,
                                  param1=100, param2=18,
                                  minRadius=34, maxRadius=43)
    # coordinates and radius of all circles detected
    print(circleshed)

    for circle in circleshed[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        # print(circle[0])
        # print('------------------------')
        # print(circle[1])
        # print('-------------------------')
        # print(circle[2])
        # print('--------------------------')
        # draw circles detected by Hough Transform
        # draw_circle = cv2.circle(img, (x, y), r, (255, 255, 255), 1, 10, 0)  # 画出检测到的圆，（255,255,255）代表白色
        # cv2.imshow("circle",draw_circle)
        # cv2.waitKey(0)
    # use K-means algorithm to group the circles and extract the minimum from each groups
    km = KMeans(n_clusters=4).fit(circleshed[0, :, 0:2])
    df = pd.DataFrame({'X': circleshed[0, :, 0],
                       'Y': circleshed[0, :, 1],
                       'Radius': circleshed[0, :, 2],
                       'Label': km.labels_}
                      )
    mins = df.sort_values('Y', ascending=False).groupby('Label', as_index=False).first()
    print(mins)
    for index, i in mins.iterrows():
        center = (i['X'].astype("int"), i['Y'].astype("int"))
        x_sum = x_sum + i['X']
        y_sum = y_sum + i['Y']
        # x=x+i['X']
        # y=y+i['Y']
        cv2.circle(hed, center, 1, (255, 0, 255), 8)
        radius = i['Radius'].astype("int")
        cv2.circle(hed, center, radius, (255, 0, 255), 8)
    cv2.imshow("four circles", hed)
    cv2.waitKey(0)
    # show the circles
    # for index, i in mins.iterrows():
    #     center = (i['X'].astype("int"), i['Y'].astype("int"))
    #     cv.circle(canny, center, 1, (255, 255, 255), 3)
    #     radius = i['Radius'].astype("int")
    #     cv.circle(canny, center, radius, (255, 255, 255), 3)
    # cv.imshow("canny", canny)
    # cv.waitKey()
    # x_ave = sum(mins.iterrows['X']) / 4
    # y_ave = sum(mins.iterrows['Y']) / 4
    # diff_x=center[0]-x_ave
    # diff_y=center[1]-y_ave
    # diff_pix=math.sqrt(diff_x*diff_x+diff_y*diff_y)
    # print(diff_pix,diff_x,diff_y)
    # v = corners[12][0] - corners[11][0]
    # d = math.sqrt(v[0] * v[0] + v[1] * v[1])
    print('measured center: ', x_sum / 4, y_sum / 4)
    x_diff = x_sum / 4 - center_x
    y_diff = y_sum / 4 - center_y
    diff = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    # difference between measured center and actual center
    print('difference: ', diff)
    return (mins['X'].mean() - W / 2, mins['Y'].mean() - H / 2)


# # function to call local_grids_2
def detect_bolts_2():
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cv2.dnn_registerLayer("Crop", CropLayer)
    cnt = 0
    img = cv2.imread("after_close/AF_9.jpg")
    x, y = img.shape[0:2]
    ret, corners = cv2.findChessboardCorners(img, (11, 8), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                             cv2.CALIB_CB_FAST_CHECK +
                                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    v = corners[12][0] - corners[13][0]
    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
    ratio = d / 15
    print('ratio: ', ratio)
    # resize image to actual size
    img = cv2.resize(img, (int(x / ratio), int(y / ratio)))
    local_grid_2(img, net)


if __name__ == '__main__':
    # call your function here to test
    detect_base()
