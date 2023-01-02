import cv2
from sklearn.cluster import KMeans
import pandas as pd
import math
import numpy as np
import glob
# test base_grid and base_grid2


NN_FLAG = False  # Select whether to use neural network to perform edge detection of base
OLD_BASE_FLAG = False  # Demonstrate old method of finding bases
CANNY_PARAMETER = (100, 150)


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

# def HED(image, net):
#     (H, W) = image.shape[:2]
#     # image = cv2.resize(image, (downW, downH))
#     # 根据输入图像为全面的嵌套边缘检测器（Holistically-Nested Edge Detector）构建一个输出blob
#     blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(H, W),
#                                 mean=(104.00698793, 116.66876762, 122.67891434),
#                                 swapRB=False, crop=False)
#
#     # # 设置blob作为网络的输入并执行算法以计算边缘图
#     print("[INFO] performing holistically-nested edge detection...")
#     net.setInput(blob)
#     hed = net.forward()
#     # 调整输出为原始图像尺寸的大小
#     hed = hed[0, 0]
#     # hed = cv2.resize(hed[0, 0], (W, H))
#     # 将图像像素缩回到范围[0,255]并确保类型为“UINT8”
#     hed = (255 * hed).astype("uint8")
#     return hed

def direction(dst):
    direction = 0  # This is to ensure the robustness of default case
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    image = dst
    size = image.shape
    gray1 = cv2.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray1, (5, 5), None)
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
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        if thresh[ur[0], ur[1]] > 0 and thresh[ll[0], ll[1]] == 0:
            direction = -90
        elif thresh[lr[0], lr[1]] > 0 and thresh[ul[0], ul[1]] == 0:
            direction = 180
        elif thresh[ll[0], ll[1]] > 0 and thresh[ur[0], ur[1]] == 0:
            direction = 90
    # print('direction',str(direction))
    return direction

######################################################################################
def base_X(image, net):
    print("baseX start")
    # 找棋盘格的中心
    size = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(
        image[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
        cv2.COLOR_BGR2GRAY)
    print("gray complete")
    ret, corners = cv2.findChessboardCorners(gray1, (5, 5), None)
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if NN_FLAG:
        hed = HED(image, net, downW=400, downH=400)  # 如果无法识别请改用神经网络
    else:
        hed = cv2.Canny(cv2.resize(blurred, (800, 600)), CANNY_PARAMETER[0], CANNY_PARAMETER[1])
    center = corners[12][0]
    hed = cv2.resize(hed, (W, H))
    # 找闭合轮廓
    contours, hier = cv2.findContours(hed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("start finding min")
    for cnt in contours:
        print("start cnt")
        # 筛出小轮廓，找到地基轮廓
        if cv2.contourArea(cnt) > H * W * 0.2:
            print("cnt calculated")
            M = cv2.moments(cnt)
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
        image = cv2.imread(fname)
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
    dir = direction(cv2.imread(images[idmin]))
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
    circleshed = cv2.HoughCircles(hed, cv2.HOUGH_GRADIENT, 1, round(H / 2),
                                 param1=100, param2=50,
                                 minRadius=round(H / 4), maxRadius=round(H / 2))
    return circleshed[0, 0, 0] - W / 2 - corr, circleshed[0, 0, 1] - H / 2 - corr


def get_camera_matrix(images, criteria):
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    file1 = images[0]
    img_1 = cv2.imread(file1)
    size = img_1.shape
    h, w = img_1.shape[:2]
    gray_full = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    size_full = gray_full.shape[::-1]
    for fname in images:
        img = cv2.imread(fname)
        gray_crop = cv2.cvtColor(
            img[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
            cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_crop, (11, 8), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                      cv2.CALIB_CB_FAST_CHECK +
                                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print("find chessboard corners!")
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray_crop, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if len(corners2):
                for each in corners2:
                    each[0][1] += math.ceil(size[0] / 3)
                    each[0][0] += math.ceil(size[1] / 4)
                img_points.append(corners2)
            else:
                for each in corners:
                    each[0][1] += math.ceil(size[0] / 3)
                    each[0][0] += math.ceil(size[1] / 4)
                img_points.append(corners)
            print("find corner_sub_pix!")
    print(len(img_points[0]))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size_full, None, None)
    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    dist[0][4] = 0
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数
    print("-----------------------------------------------------")
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
    return newcameramtx, mtx, dist

def perspective_recover(dst, criteria, w, h, size=4000, ratio=0.03):
    # recover the perspective of the picture, i.e. make the chessborad in the z=0 surface.
    # - size: 导出图片的分辨率
    # - ratio：相当于棋盘格上一格占图片宽度的比例
    print("h:", h)
    print("w:", w)
    size = dst.shape
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(
    #     dst[math.ceil(size[0] / 3):math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4):math.ceil(3 * size[1] / 4)],
    #     cv2.COLOR_BGR2GRAY)
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
    # for each in corners:
    #     each[0][1] += math.ceil(size[0] / 3)
    #     each[0][0] += math.ceil(size[1] / 4)
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


def test_direct_hough_with_hed(image, net):
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
        center_x = i[0]
        center_y = i[1]
    return center_x, center_y
    # x, y = hed.shape[0:2]
    # hed = cv2.resize(hed, (int(y / 6), int(x / 6)))
    # cv2.imshow("hed", hed)
    # cv2.imwrite("new_hed_1.jpg", hed)
    # x, y = image.shape[0:2]
    # image = cv2.resize(image, (int(y / 6), int(x / 6)))
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test_hed_or_canny_with_find_contour(image, net):
    (H, W) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # hed = HED(image, net, downW=400, downH=400)  # 如果无法识别请改用神经网络
    hed = cv2.Canny(cv2.resize(blurred, (800, 600)), CANNY_PARAMETER[0], CANNY_PARAMETER[1])
    # cv2.imshow("hed", hed)
    # cv2.imwrite("old_canny_1.jpg", hed)
    contours, hier = cv2.findContours(hed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("start finding min")
    for cnt in contours:
        print("start cnt")
        if cv2.contourArea(cnt) > H * W * 0.2:
            temp = np.ones(hed.shape, np.uint8) * 255
            # 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
            cv2.drawContours(temp, cnt, -1, (0, 255, 0), 3)
            print("find base!")
            cv2.imshow("contours", temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\old_model\*.jpg')
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model\base\*.jpg')
    # images = glob.glob(R'D:\Courses\IPP\images\*.jpg')
    images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model\new_base_cloth\*.jpg')
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\images\*.jpg')
    print("load the pictures!")
    newcameramtx, mtx, dist = get_camera_matrix(images_cam, criteria)
    # dst1, w, h = distort_recover(images, criteria)
    # images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\far_v2\*.jpg')
    cnt = -1
    for fname in images_cam:
        cnt += 1
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst2 = perspective_recover(dst1, criteria, w, h)
        file_name = "./new_model_perspective/AF_" + str(cnt) + ".jpg"
        cv2.imwrite(file_name, dst2)
        print("已写入:", file_name)
    # fname = images_cam[0]
    # img = cv2.imread(fname)
    # h, w = img.shape[:2]
    # dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # cv2.imwrite("undistort_new_model_1.jpg", dst1)
    # dst2 = perspective_recover(dst1, criteria, w, h)
    # cv2.imwrite("new_model_1.jpg", dst2)

def main_2():
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cv2.dnn_registerLayer("Crop", CropLayer)
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model\base\*.jpg')
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\old_model\*.jpg')
    # images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model\new_base_cloth\*.jpg')
    images_cam = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model_perspective\*.jpg')
    fname = images_cam[0]
    img = cv2.imread(fname)
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 24), int(x / 24)))
    test_direct_hough_with_hed(img, net)
    # center_x, center_y = test_direct_hough_with_hed(img, net)
    # test_hed_or_canny_with_find_contour(img, net)

def getDist_P2P(Point0,PointA):
    distance = math.pow((Point0[0]-PointA[0]), 2) + math.pow((Point0[1]-PointA[1]),2)
    distance = math.sqrt(distance)
    return distance


# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     points = []
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%f,%f" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         points.append([x, y])
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#         h, w = img.shape[:2]
#         print("the original circle center: ", x, y)
#         print("the hough circle center: ", center_x, center_y)
#         x_diff = abs(center_x - x) / w * 100
#         y_diff = abs(center_y - y) / h * 100
#         print("the difference percentage of the pic size is: ", "{}%".format(x_diff), "{}%".format(y_diff))
#         string_1 = "the original circle center: {}, {}".format(x, y)
#         file_handle.write(string_1)
#         string_2 = "the hough circle center: {}, {}".format(center_x, center_y)
#         file_handle.write(string_2)
#         string_3 = "the difference percentage of the pic size is: {}%, {}%".format(x_diff, y_diff)
#         file_handle.write(string_3)
#         file_handle.write("\n")
#         file_handle.flush()
#         print("one pic is done!")
#         # sys.exit(0)
#         # return
#
#

def calculate_grid_length_stdev(corners):
    grid_length = []
    for i in range(0, 10):
        grid_length.append(getDist_P2P(corners[i][0], corners[i+1][0]))
    for i in range(77, 87):
        grid_length.append(getDist_P2P(corners[i][0], corners[i+1][0]))
    for i in range(0, 7):
        grid_length.append(getDist_P2P(corners[11*i][0], corners[11*(i+1)][0]))
        grid_length.append(getDist_P2P(corners[11*i+10][0], corners[11*(i+1)+10][0]))
    print(np.std(grid_length, ddof=1))
    return np.std(grid_length, ddof=1)


criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
# images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\new_model_perspective\*.jpg')
images = glob.glob(R'D:\Courses\SJTU-IPP-Program2022\serverV2_new\tt\*.jpg')
file_handle = open('center.txt', mode='w')
protoPath = "deploy.prototxt"
modelPath = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
cv2.dnn_registerLayer("Crop", CropLayer)
cnt = 0
for fname in images:
    img = cv2.imread(fname)
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 6), int(x / 6)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    points = [corners[0][0], corners[1][0], corners[38][0]]
    cnt += 1
    center_x, center_y = test_direct_hough_with_hed(img, net)
    hough_center = [center_x, center_y]
    len_of_side = getDist_P2P(points[0], points[1])
    diff_center = getDist_P2P(points[2], hough_center)
    diff_in_mm = diff_center/len_of_side * 15
    std_ev = calculate_grid_length_stdev(corners)
    print("the original circle center: ", round(points[2][0], 2), round(points[2][1], 2))
    print("the hough circle center: ", round(center_x, 2), round(center_y, 2))
    print("the difference in mm is: ", round(diff_in_mm, 2))
    print("grid_length_stdev: ", round(std_ev, 3))
    # string_1 = "the original circle center: {}, {}".format(round(points[2][0], 2), round(points[2][1], 2))
    # file_handle.write(string_1)
    # string_2 = "the hough circle center: {}, {}".format(round(center_x, 2), round(center_y, 2))
    # file_handle.write(string_2)
    string_3 = "the difference in mm is: , {}mm".format(round(diff_in_mm, 2))
    file_handle.write(string_3)
    string_4 = "the standard deviation is:{} pixels".format(round(std_ev, 3))
    file_handle.write(string_4)
    file_handle.write("\n")
    file_handle.flush()
    cv2.circle(img, (int(points[2][0]), int(points[2][1])), 2, (0, 255, 0), 3)
    file_name = "./new_white_base_model/new_model_perspective_circle_" + str(cnt) + ".jpg"
    cv2.imwrite(file_name, img)
    # cv2.resizeWindow("image", 1740, 1227)
    img = cv2.resize(img, (int(y / 6), int(x / 6)))
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print("one pic is done!")


