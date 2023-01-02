import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
#objp = np.zeros((11 * 13, 3), np.float32)
#objp[:, :2] = np.mgrid[0:13, 0:11].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp = np.zeros((5 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2)

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob(R'C:\Users\dell\Desktop\Courses\IPP\serverV2\chessboard4*.jpg')
print("load the pictures!")
i=0;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    #ret, corners = cv2.findChessboardCorners(gray, (13, 11), None)
    ret, corners = cv2.findChessboardCorners(gray, (5, 5), None)
    #print(corners)
    print("find chessboard corners!")
    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        #cv2.drawChessboardCorners(img, (13, 11), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.drawChessboardCorners(img, (5, 5), corners, ret)
        i+=1;
        #cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.imwrite('conimg_4_' + str(i) + '.jpg', img)
        cv2.waitKey(1500)
    if i == 3:
        print("processing image3")
        h, w = img.shape[:2]
        objp2 = np.zeros((5 * 5, 2), np.float32)
        cap = max(h, w)
        k = np.mgrid[-2:3:1, -2:3:1].T.reshape(-1, 2)
        #print(k)
        ratio = 0.04
        objp2[:, :2] = (k * ratio + 0.5) * cap
        #print(objp)
        #print(objp2)
        if [corners2]:
            pts1 = np.float32([corners2[0][0], corners2[4][0], corners2[-1][0]])
            vec = corners2[4][0] - corners2[0][0]
            tan = vec[1] / vec[0]
        else:
            pts1 = np.float32([corners[0][0], corners[4][0], corners[-1][0]])
            vec = corners[4][0] - corners[0][0]
            tan = vec[1] / vec[0]
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
        M = cv2.getAffineTransform(pts1, pts2)
        img3 = cv2.warpAffine(img, M, (cap, cap))
        size2 = 800
        img3 = cv2.resize(img3, (size2, size2))
print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")
# img = cv2.imread(images[2])
img = img3
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
print("------------------使用undistort函数-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
#cv2.imwrite('calibresult3.jpg', dst1)
cv2.imwrite('calibresult4.jpg', dst1)
print ("方法一:dst的大小为:", dst1.shape)




