import cv2
import numpy as np
import glob


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
    #pbj2:[1000. 1200.]
         # [1200. 1200.]
         # [1400. 1200.]
         # [1600. 1200.]
         # [1800. 1200.]
         # [2000. 1200.]
         # [2200. 1200.]
         # [2400. 1200.]
         # [2600. 1200.]
         # [2800. 1200.]
         # [3000. 1200.]
         # [1000. 1400.]
         # [1200. 1400.]...
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


def distort_recover(images, criteria):
    # obtain the inner matrix of the camera and undistort the image.
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    size_all = 0
    file1 = images[0]
    img_1 = cv2.imread(file1)
    h_, w_ = img_1.shape[:2]
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    size_all = gray_1.shape[::-1]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print("find chessboard corners!")
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            print("find corner_sub_pix!")
            # cv2.imwrite('conimg'+str(i)+'.jpg', img)
            # cv2.imwrite('conimg_6_' + str(i) + '.jpg', img)
            # cv2.waitKey(1500)
    print(len(img_points))
    # cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size_all, None, None)
    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数
    print("-----------------------------------------------------")
    img = cv2.imread(images[1])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
    print(newcameramtx)
    print("------------------使用undistort_perspective函数-------------------")
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # x, y, w, h = roi
    # dst1 = dst[y:y + h, x:x + w]
    return dst, w_, h_


if __name__ == '__main__':
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    images = glob.glob(R'D:\Courses\IPP\serverV2\chessboard_1_8\*.jpg')
    # images = glob.glob(R'D:\Courses\IPP\images\*.jpg')
    print("load the pictures!")
    dst1, w, h = distort_recover(images, criteria)
    dst2 = perspective_recover(dst1, criteria, w, h)
    # dst2 = perspective_recover(img, criteria)
    cv2.imwrite('c_test.jpg', dst2)
    print ("方法一:dst的大小为:", dst2.shape)

