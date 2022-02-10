import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import zipfile
import numpy as np
import sys
import cv2 as cv
import glob
import os
from sklearn.cluster import KMeans
import pandas as pd
import math

NUMBERS = 2  # Number of images used to detect bolts direction
NN_FLAG = False  # Select whether to use neural network to perform edge detection of base
OLD_BASE_FLAG = False  # Demonstrate old method of finding bases
CANNY_PARAMETER = (100, 150)

FAILURE_FLAG = False


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


def local_grid(gray):
    # detect the center of bolts relative to the centre of
    (H, W) = gray.shape[:2]
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 30, 150)
    # obtain edge through Canny alogrithm
    circleshed = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 20,
                                 param1=100, param2=16,
                                 minRadius=20, maxRadius=50)
    # use K-means algorithm to group the circles and extract the minimum from each groups
    km = KMeans(n_clusters=4).fit(circleshed[0, :, 0:2])
    df = pd.DataFrame({'X': circleshed[0, :, 0],
                       'Y': circleshed[0, :, 1],
                       'Radius': circleshed[0, :, 2],
                       'Label': km.labels_}
                      )
    mins = df.sort_values('Y', ascending=False).groupby('Label', as_index=False).first()
    # show the circles
    # for index, i in mins.iterrows():
    #     center = (i['X'].astype("int"), i['Y'].astype("int"))
    #     cv.circle(canny, center, 1, (255, 255, 255), 3)
    #     radius = i['Radius'].astype("int")
    #     cv.circle(canny, center, radius, (255, 255, 255), 3)
    # cv.imshow("canny", canny)
    # cv.waitKey()
    return (mins['X'].mean() - W / 2, mins['Y'].mean() - H / 2)


def HED(image, net, downW=800, downH=800):
    (H, W) = image.shape[:2]
    image = cv.resize(image, (downW, downH))
    # 根据输入图像为全面的嵌套边缘检测器（Holistically-Nested Edge Detector）构建一个输出blob
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(800, 800),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=True)

    # # 设置blob作为网络的输入并执行算法以计算边缘图
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    # 调整输出为原始图像尺寸的大小
    hed = cv.resize(hed[0, 0], (W, H))
    # 将图像像素缩回到范围[0,255]并确保类型为“UINT8”
    hed = (255 * hed).astype("uint8")
    return hed


######################################################################################
def base_X(image, net):
    print("baseX start")
    # 找棋盘格的中心
    size = image.shape
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv.COLOR_BGR2GRAY)
    print("gray complete")
    ret, corners = cv.findChessboardCorners(gray1, (5, 5), None)
    # print(corners)
    for each in corners:
        each[0][1] += math.ceil(size[0] / 3)
        each[0][0] += math.ceil(size[1] / 4)
    print("find corners")
    if ret == False:
        return 0, 100
    print("judge direction")
    # 判断棋盘格与相机光轴的相对方向
    vd1 = corners[2][0] - corners[-3][0]
    vd2 = corners[10][0] - corners[14][0]
    dir = min(abs(vd1[0] / vd1[1]), abs(vd2[0] / vd2[1]))
    # 判断棋盘格的单位长度
    v = corners[12][0] - corners[13][0]
    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
    # 边缘识别
    (H, W) = image.shape[:2]
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    if NN_FLAG:
        hed = HED(image, net, downW=400, downH=400)  # 如果无法识别请改用神经网络
    else:
        hed = cv.Canny(cv.resize(blurred, (800, 600)), CANNY_PARAMETER[0], CANNY_PARAMETER[1])
    center = corners[12][0]
    hed = cv.resize(hed, (W, H))
    # 找闭合轮廓
    contours, hier = cv.findContours(hed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("start finding min")
    for cnt in contours:
        print("start cnt")
        # 筛出小轮廓，找到地基轮廓
        if cv.contourArea(cnt) > H * W * 0.2:
            print("cnt calculated")
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # # Show the center detection
            # cv.circle(hed, (int(center[0]), int(center[1])), 20, (255, 255, 255), 3)
            # cv.circle(hed, (cx,cy), 20, (255, 255, 255), 3)
            # cv.imshow("hed", cv.resize(hed, (800, 600)))
            # cv.waitKey()
            l = abs(cx - center[0])
            return l / d * 32, dir
    return 0, 100


def base_grid2(images, net):
    print("base_grid2 start")
    ids = []
    lens = []
    dirs = []
    for index, fname in enumerate(images):
        image = cv.imread(fname)
        l, d = base_X(image, net)
        ids.append(index)
        lens.append(l)
        dirs.append(d)
    print("index and fname complete")
    if (np.array(dirs) == 100).all():
        global FAILURE_FLAG
        FAILURE_FLAG = True
        print("Detection failure, please consider use Neural Network/Old Method for base detection!")
        return 0, 0
    print("start kMeans")
    km = KMeans(n_clusters=2).fit(np.array(lens).reshape(-1, 1))
    print("end kMeans\nstart df")
    df = pd.DataFrame({'index': ids,
                       'l': lens,
                       'dir': dirs,
                       'label': km.labels_
                       })
    df = df[((df['l'] - df['l'].mean()) / df['l'].std()) < 1]  # Remove outliers that are too large
    print("end df")
    # print(df.to_string())
    # 寻找棋盘格最接近垂直/水平的帧
    print("start to find min or max")
    minf = df[df.dir == df.dir.min()]
    lmin = minf['l'].values[0]
    idmin = int(minf['index'].values)
    # 用聚类方法减少寻找最大值过程中产生的偏差
    lmax = df.groupby('label', as_index=False)[["l"]].mean().max().values[1]
    ret = (lmin, math.sqrt(abs(lmax * lmax - lmin * lmin)))
    print("found min or max")
    # 判断这一帧的方向
    print("start judge")
    dir = direction(cv.imread(images[idmin]))
    print("end judge")
    if dir == 90:
        ret = (ret[1], -ret[0])
    elif dir == 180:
        ret = (-ret[0], -ret[1])
    elif dir == -90:
        ret = (-ret[1], ret[0])
    return ret


def base_grid(image, net, corr=90):
    # detect the center of round base
    # The result of HED is skewed from original picture, corr is a corrective item
    hed = HED(image, net)
    # HED seems to have some blur
    (H, W) = hed.shape[:2]
    circleshed = cv.HoughCircles(hed, cv.HOUGH_GRADIENT, 1, round(H / 2),
                                 param1=100, param2=50,
                                 minRadius=round(H / 4), maxRadius=round(H / 2))
    # show the circle
    # df = pd.DataFrame({'X': circleshed[0, :, 0],
    #                    'Y': circleshed[0, :, 1],
    #                    'Radius': circleshed[0, :, 2]}
    #                   )
    # for index, i in df.iterrows():
    #     center = (i['X'].astype("int"), i['Y'].astype("int"))
    #     cv.circle(hed, center, 1, (255, 255, 255), 3)
    #     radius = i['Radius'].astype("int")
    #     cv.circle(hed, center, radius, (255, 255, 255), 3)
    # cv.circle(hed, (round(W/2 + corr),round(H/2 + corr)), 1, (255, 255, 255), 3)
    # cv.imshow("hed", cv.resize(hed, (800, 800)))
    # cv.waitKey()
    return circleshed[0, 0, 0] - W / 2 - corr, circleshed[0, 0, 1] - H / 2 - corr


def square_grid_NN(image, net):
    # detect center of squared base
    # ! Notice this method is untested
    hed = HED(image, net)
    (H, W) = hed.shape[:2]
    contours, hier = cv.findContours(hed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > H * W * 0.2:  # remove small areas like noise etc
            hull = cv.convexHull(cnt)  # find the convex hull of contour
            hull = cv.approxPolyDP(hull, 0.1 * cv.arcLength(hull, True), True)
            if len(hull) == 4:
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center = (cx - W / 2, cy - W / 2)
                return center
    return 0, 0  # default value if unable to find center


def bwareaopen(img, th):
    ## Helper function for removing smaller
    cnts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < th:
            cv.drawContours(img, [c], -1, (0, 0, 0), -1)

    # Morph close
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)
    return close


def square_grid_normal(image):
    # detect center of squared base
    # ! Notice this is only tried on ONE sample picture is provided
    dst = image[:, :, 0]
    thresh = cv.threshold(dst, 255 * 0.4, 255, cv.THRESH_BINARY)[1]
    h, w = image.shape[:2]
    th = round(h * w * 0.1)

    close = bwareaopen(thresh, th)
    kernel = np.ones((5, 5), np.uint8)
    close = bwareaopen(255 - cv.dilate(close, kernel, iterations=1), th)
    mask = bwareaopen(255 - cv.dilate(close, kernel, iterations=1), th)

    contours, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > 2 * th:  # remove small areas like noise etc
            hull = cv.convexHull(cnt)  # find the convex hull of contour
            hull = cv.approxPolyDP(hull, 0.1 * cv.arcLength(hull, True), True)
            if len(hull) == 4:
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center = (cx - w / 2, cy - w / 2)
                return center
    return 0, 0  # default value if unable to find center


def direction(dst):
    direction = 0  # This is to ensure the robustness of default case
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    image = dst
    size = image.shape
    gray1 = cv.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray1, (5, 5), None)
    for each in corners:
        each[0][1] += math.ceil(size[0] / 3)
        each[0][0] += math.ceil(size[1] / 4)
    # Center of each black square to locate white holes indicating location
    if ret:
        ul = ((corners[0][0] + corners[1][0] + corners[5][0] + corners[6][0]) / 4).astype(int)
        ur = ((corners[3][0] + corners[4][0] + corners[8][0] + corners[9][0]) / 4).astype(int)
        ll = ((corners[-5][0] + corners[-4][0] + corners[-9][0] + corners[-10][0]) / 4).astype(int)
        lr = ((corners[-1][0] + corners[-2][0] + corners[-6][0] + corners[-7][0]) / 4).astype(int)
        # NOTICE: The indicated angle is NOT the actual angle by the viewer, but a 180 degree rotated one due to Opencv coding format
        # Adding the following line change it to the real angle， but result in the rotation of coordinates given by program

        # (ul, ur, ll, lr) = (lr, ll, ur, ul)
        thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)[1]
        if thresh[ur[0], ur[1]] > 0 and thresh[ll[0], ll[1]] == 0:
            direction = -90
        elif thresh[lr[0], lr[1]] > 0 and thresh[ul[0], ul[1]] == 0:
            direction = 180
        elif thresh[ll[0], ll[1]] > 0 and thresh[ur[0], ur[1]] == 0:
            direction = 90
    # print('direction',str(direction))
    return direction


def obthomography(dst, criteria, size=800, ratio=0.04):
    # Obtain homography of the picture centred at the centre of chessboard
    h, w = dst.shape[:2]
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    '''image = dst
    size = image.shape
    gray1 = cv.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv.COLOR_BGR2GRAY)'''
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (5, 5), None)
    '''for each in corners:
        if each is not None:
            each[0][1] += math.ceil(w / 3)
            each[0][0] += math.ceil(h / 4)'''
    # corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # uncomment this line to perform subpixel detection of corner, significantly slow the speed
    # prepare corresponding points for homography
    objp2 = np.zeros((5 * 5, 2), np.float32)
    cap = max(h, w)
    k = np.mgrid[-2:3:1, -2:3:1].T.reshape(-1, 2)
    objp2[:, :2] = (k * ratio + 0.5) * cap
    pts1 = np.float32([corners[0][0], corners[4][0], corners[-1][0]])

    # rectify picture to proper orientation that make the bolts face downward
    vec = corners[4][0] - corners[0][0]
    tan = vec[1] / vec[0]
    # !! only basic cases considered !!
    if tan <= 1 and tan >= -1:
        if vec[0] >= 0:
            pts2 = np.float32([objp2[0], objp2[4], objp2[-1]])
        else:
            pts2 = np.float32([objp2[-1], objp2[-5], objp2[0]])
    else:
        if vec[1] >= 0:
            pts2 = np.float32([objp2[4], objp2[-1], objp2[-5]])
        else:
            pts2 = np.float32([objp2[-5], objp2[0], objp2[4]])

    # obtain and perform transformation
    M = cv.getAffineTransform(pts1, pts2)
    # H = cv.findHomography(corners, objp2)
    dst = cv.warpAffine(dst, M, (cap, cap))
    dst = cv.resize(dst, (size, size))
    return dst


def processimage(path):
    zFile = zipfile.ZipFile(path, "r")
    for fileM in zFile.namelist():
        zFile.extract(fileM, "./process")
    zFile.close()
    os.remove(path)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('./process/*.jpg')
    print("calibrating cameras...")
    gray1 = []
    gray = []
    for fname in images:
        print("start:" + fname)
        img = cv.imread(fname)
        size = img.shape
        '''img = img[math.ceil(size[0] / 5): math.ceil(4 * size[0] / 5),
              math.ceil(size[1] / 5): math.ceil(4 * size[1] / 5)]'''
        # cv.imwrite(fname, img)
        '''temp_img = cv.imread(fname)
        print("start to cut the pictures")
        img = temp_img[85:400, 85:400]
        cv.imshow("picture", img)'''
        print("finish:" + fname)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray1 = cv.cvtColor(
            img[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
            cv.COLOR_BGR2GRAY)
        print("binary color finished" + fname)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray1, (5, 5), None)
        print("findchesscorners finished:" + fname)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("ret is true")
            objpoints.append(objp)
            corners = cv.cornerSubPix(gray1, corners, (11, 11), (-1, -1), criteria)
            for each in corners:
                each[0][1] += math.ceil(size[0] / 3)
                each[0][0] += math.ceil(size[1] / 4)
            # print(corners)
            print("subpix complete:" + fname)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dist = dist[0:4]  # 最后一项会让图片扭曲
    loc_grids = []
    base_grids = []
    diff = []
    diff2 = []
    print("[INFO] loading edge detector for base recognition...")
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv.dnn.readNetFromCaffe(protoPath, modelPath)
    print("Processing images...")
    n2 = len(images) - 1
    if NN_FLAG:
        n2 = min(5, n2)
    basecenter2 = base_grid2(images[0:n2], net)  # 提取所有图片中地基的横坐标，用统计学方式判断出实际的横纵坐标，速度快且效果更好
    print("find base center 2")
    for image in images[0:NUMBERS]:  # 选择处理的张数
        print(image)
        img = cv.imread(image)
        h, w = img.shape[:2]
        print("start to find the optimal")
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print("found the optimal")
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        dst_bolts = obthomography(dst, criteria)  # for bolts
        gray_bolts = dst_bolts[:, :, 0] * 0.5 + dst_bolts[:, :, 1] * 0.5 + -dst_bolts[:, :, 2] * 0.6
        gray_bolts = gray_bolts.astype(np.uint8)
        loc = local_grid(gray_bolts)
        dir = direction(dst_bolts)  # clock wise angle!
        if dir == 90:
            loc = (loc[1], -loc[0])
        elif dir == 180:
            loc = (-loc[0], -loc[1])
        elif dir == -90:
            loc = (-loc[1], loc[0])
        loc_grids.append(loc)
        # the coordination is unified to assume each grid on chessboard takes 800 * 0.04 = 32 units of length
        # NOTICE: A positive value indicates lower and righter, while a negative value indicates upper and lefter
        # 寻找地基中心的第一个方法,似乎并不准确
        if OLD_BASE_FLAG:
            roundbase = obthomography(img, criteria, size=2000, ratio=0.016)
            base = base_grid(roundbase, net)
            base = (base[0], base[1])
            if dir == 90:
                base = (base[1], -base[0])
            elif dir == 180:
                base = (-base[0], -base[1])
            elif dir == -90:
                base = (-base[1], base[0])
            base_grids.append(base)
            diff.append((loc[0] - base[0], loc[1] - base[1]))
        diff2.append((loc[0] - basecenter2[0], loc[1] - basecenter2[1]))
        # diff is the vector from center of base to center of bolts
    with open(os.path.join(sys.path[0], "Receive", "coord.txt"), 'a') as text_file:
        print("bolt center: ", file=text_file)
        print(loc_grids, file=text_file)
        if OLD_BASE_FLAG:
            print("\nbase center(old method): ", file=text_file)
            print(base_grids, file=text_file)
            print("\nlocation(old method): ", file=text_file)
            print(diff, file=text_file)
        global FAILURE_FLAG
        if FAILURE_FLAG:
            print("Detection failure, please consider use Neural Network/Old Method for base detection!",
                  file=text_file)
            FAILURE_FLAG = False
        print("\nbase center: ", file=text_file)
        print(basecenter2, file=text_file)
        print("\nlocation: ", file=text_file)
        print(diff2, file=text_file)
    print("process finished")
    for fname in images:
        os.remove(fname)


def on_created(event):
    time.sleep(1)
    print(f"{event.src_path} created")
    if zipfile.is_zipfile(event.src_path):
        processimage(event.src_path)
    else:
        return


def on_deleted(event):
    if zipfile.is_zipfile(event.src_path):
        print(f"{event.src_path} deleted")
    else:
        return


def on_modified(event):
    if zipfile.is_zipfile(event.src_path):
        print(f"{event.src_path} modified")
    else:
        return
    processimage(event.src_path)


def on_moved(event):
    print(f"{event.src_path} moved to {event.dest_path}")


watch_patterns = ["*.zip"]  # 监控文件的模式, 必须为列表形式
ignore_patterns = None  # 设置忽略的文件模式
ignore_directories = False  # 是否忽略文件夹变化
case_sensitive = True  # 是否对大小写敏感
event_handler = PatternMatchingEventHandler(watch_patterns, ignore_patterns, ignore_directories, case_sensitive)

watch_path = ".\Receive"  # 监控目录
go_recursively = True  # 是否监控子文件夹
my_observer = Observer()
my_observer.schedule(event_handler, watch_path, recursive=go_recursively)

event_handler.on_created = on_created
event_handler.on_deleted = on_deleted
event_handler.on_modified = on_modified
event_handler.on_moved = on_moved

if __name__ == '__main__':
    print("server start")
    my_observer.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
