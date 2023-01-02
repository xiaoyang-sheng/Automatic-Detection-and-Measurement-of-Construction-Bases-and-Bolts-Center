import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread('102124169191958027.jpg', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
cv2.imwrite('th2.jpg', th2)
cv2.imwrite('th3.jpg', th3)
#for i in range(3):
    #plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    #plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    #plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    #plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#plt.show()


# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
#     detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
#     dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo', dst)
#
#
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3
#
# img2 = cv2.imread('test_save.jpg')
# gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# cv2.namedWindow('canny demo')
#
# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
#
# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()


img3 = cv2.GaussianBlur(th2, (5, 5), 0)
img4 = cv2.GaussianBlur(img, (5, 5), 0)
img5 = cv2.GaussianBlur(th3, (5, 5), 0)
canny = cv2.Canny(img3, 50, 550)
canny2 = cv2.Canny(img4, 50, 550)
canny3 = cv2.Canny(img5, 50, 550)
cv2.imwrite('edge_th1.jpg', canny)
cv2.imwrite('edge_ori.jpg', canny2)
cv2.imwrite('edge_th3.jpg', canny3)
cv2.imwrite('gaussian_img.jpg', img4)


# cv2.imshow('Canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_2 = cv2.imread('circle.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3, 3))
cv2.imwrite('gray_blurred.jpg', gray_blurred)

circles2 = cv2.HoughCircles(img4, cv2.HOUGH_GRADIENT, 1, 15, 550, 500, 500, 2000)
print(circles2)
# cv2.HoughCircles(canny3, circles2, cv2.HOUGH_GRADIENT, 1, 100, 110, 55, 0, 0)
# circles2 = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=200)
# circles2 = cv2.HoughCircles(canny3, cv2.HOUGH_GRADIENT_ALT, 1, 30, param1=300, param2=0.65, minRadius=150)
# if circles2 is not None:  # 如果识别出圆
#     for circle in circles2[0]:
#         #  获取圆的坐标与半径
#         i = circle
#         cv2.circle(img4, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 4)
#         cv2.imwrite('circle_th3_res.jpg', img4)
#         # cv2.imshow("img_cir", canny3)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
# else:
#     # 如果识别不出，显示圆心不存在
#     print('no circle')
# cv2.destroyAllWindows()

# if circles2 is not None:
#
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(circles2))
#
#     for pt in detected_circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#
#         # Draw the circumference of the circle.
#         cv2.circle(img_2, (pt[0], pt[1]), pt[2], (0, 255, 0), 4)
#
#         # # Draw a small circle (of radius 1) to show the center.
#         # cv2.circle(img_2, (a, b), 1, (0, 0, 255), 3)
#         cv2.imshow("Detected Circle", img_2)
#         cv2.waitKey(0)
# else:
#     print('no circle')
