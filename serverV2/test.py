import cv2 as cv
import numpy as np
import glob
import time
import pandas as pd
from sklearn.cluster import KMeans
import math

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
    print(circleshed)
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
        cv.circle(canny, center, 1, (255, 0, 255), 8)
        radius = i['Radius'].astype("int")
        cv.circle(canny, center, radius, (255, 0, 255), 8)
    x, y = canny.shape[0:2]
    canny = cv.resize(canny, (int(y / 6), int(x / 6)))
    cv.imshow("canny", canny)
    cv.waitKey()
    print(mins['X'][0])
    # show the circles
    # for index, i in mins.iterrows():
    #     center = (i['X'].astype("int"), i['Y'].astype("int"))
    #     cv.circle(canny, center, 1, (255, 255, 255), 3)
    #     radius = i['Radius'].astype("int")
    #     cv.circle(canny, center, radius, (255, 255, 255), 3)
    # cv.imshow("canny", canny)
    # cv.waitKey()
    return (mins['X'].mean() - W / 2, mins['Y'].mean() - H / 2)


if __name__ == '__main__':
    img = cv.imread('227310620698943970.jpg')
    # print(img)
    # print(img.shape)
    # size = img.shape
    # img_2 = img[math.ceil(size[0] / 3): math.ceil(2 * size[0] / 3), math.ceil(size[1] / 4): math.ceil(3 * size[1] / 4)]
    # cv.imshow("img_2", img_2)
    # cv.waitKey()
    img_ = img[:, :, 0] * 0.5 + img[:, :, 1] * 0.5 + -img[:, :, 2] * 0.6
    img_ = img_.astype(np.uint8)
    # print(img_)
    # print(img_.shape)
    # img_ = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    local_grid(img_)
    # (H, W) = gray.shape[:2]
    # blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # canny = cv.Canny(blurred, 30, 150)
    # circleshed = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 10,
    #                              param1=100, param2=120)
    # print(circleshed)
    # print("circleshed[0, :, 0:2]: ", circleshed[0, :, 0:2])
    # km = KMeans(n_clusters=2).fit(circleshed[0, :, 0:2])
    # print(km.labels_)
    # circleshed = np.uint16(np.around(circleshed))
    # print(circleshed)
    # for i in circleshed[0, :]:  # 遍历矩阵每一行的数据
    #     cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv.imshow("img", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
