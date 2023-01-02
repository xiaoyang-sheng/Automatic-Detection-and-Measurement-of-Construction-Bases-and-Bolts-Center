import cv2
import numpy as np
import glob


def obthomography_1(dst, criteria, size=4000, ratio=0.03):
    # Obtain homography of the picture centred at the centre of chessboard
    # - size: 导出图片的分辨率
    # - ratio：相当于棋盘格上一格占图片宽度的比例
    h, w = dst.shape[:2]
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
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # uncomment this line to perform subpixel detection of corner, significantly slow the speed
    # prepare corresponding points for homography
    objp2 = np.zeros((11 * 8, 2), np.float32)
    cap =  max(h, w)
    k = np.mgrid[-5:6:1, -4:4:1].T.reshape(-1, 2)
    objp2[:, :2] = (k * ratio + 0.5) * cap
    # print(objp2)
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
    # print("corners=")
    # print(corners)
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
    # H = cv2.findHomography(corners, objp2)
    # dst = cv2.warpPerspective(dst, M, (cap, cap))
    dst = cv2.warpPerspective(dst, M, (cap, cap))
    # dst = cv2.resize(dst, (size, size))
    return dst


if __name__ == '__main__':
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)  # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    images = glob.glob(R'D:\Courses\IPP\serverV2\chessboard_1_8\f.jpg')
    img = cv2.imread(images[0])
    dst2 = obthomography_1(img, criteria)
    cv2.imwrite('calibrate_test.jpg', dst2)
    print("方法一:dst的大小为:", dst2.shape)