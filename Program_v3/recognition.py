import cv2
import numpy as np
import glob
import math

from crop import * # crop.py is custom file

SOURCE_PATH = 'result/close_noangle/perspective/*'
RESULT_PATH = 'result/close_noangle/recognition/{num}.jpg'
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "hed_pretrained_bsds.caffemodel"

# define crop class for hed
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

# HED function
def HED(image, net, downW=800, downH=800):
    (H, W) = image.shape[:2]
    image = cv2.resize(image, (downW, downH))
    # 根据输入图像为全面的嵌套边缘检测器（Holistically-Nested Edge Detector）构建一个输出blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(800, 800),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=True)
    # 设置blob作为网络的输入并执行算法以计算边缘图
    net.setInput(blob)
    hed = net.forward()
    # 调整输出为原始图像尺寸的大小
    hed = cv2.resize(hed[0, 0], (W, H))
    # 将图像像素缩回到范围[0,255]并确保类型为“UINT8”
    hed = (255 * hed).astype("uint8")
    return hed

def takeDistance(elt):
    return elt[1]

def approx(dist1, dist2):
    if abs(dist1[1]-dist2[1]) < 10.0:
        return True
    return False

def selectCircles(circles, center):
    #print(circles)
    distList = [] # [(index, distance), ...]
    for index, i in enumerate(circles[0,:]):
        distList.append(tuple((index, math.sqrt((i[0]-center[0])**2 + (i[1]-center[1])**2))))
    distList.sort(key=takeDistance)
    #print(distList)
    for i in range(0, len(distList)):
        #print(len(distList))
        if abs(distList[i][1] - distList[i+1][1]) < 10:
            #print(distList[i][1] - distList[i+1][1])
            if abs(distList[i+1][1] - distList[i+2][1]) < 10 and abs(distList[i+2][1] - distList[i+3][1]) < 10:
                return [circles[0][distList[i][0]], circles[0][distList[i+1][0]], circles[0][distList[i+2][0]], circles[0][distList[i+3][0]]]
        

def local_grid(gray, param, center):
    # detect the center of bolts relative to the centre of
    (H, W) = gray.shape[:2]
    circleshed = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=100, param2=param,
                                 minRadius=30, maxRadius=40)

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    circleshed = selectCircles(circleshed, center)
    circleshed = np.uint16(np.around(circleshed))
    #print(circleshed)
    for i in circleshed:
            center = (i[0], i[1])
            cv2.circle(result, center, 1, (0, 255, 0), 5) # draw circle center
            radius = i[2]
            cv2.circle(result, center, radius, (0, 255, 0), 5) # draw circle outline
    return result

def detect_bolts(img, net):
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
    
    size = img.shape
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    xcenter=corners[38][0][0]
    ycenter=corners[38][0][1]
    img=img[math.ceil(ycenter-(size[0]-ycenter)/2):math.ceil(ycenter+(size[0]-ycenter)/2),
        math.ceil(xcenter-(size[1]-xcenter)/2):math.ceil(xcenter+(size[1]-xcenter)/2)]
    
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_1, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    xcenter=corners[38][0][0]
    ycenter=corners[38][0][1]
    center = [xcenter, ycenter]

    gray2 = HED(img, net)
    # 求螺栓中心坐标
    result = local_grid(gray2, 18, center)
    return result

if __name__ == '__main__':
    cv2.dnn_registerLayer("Crop", CropLayer)
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    images = [cv2.imread(file) for file in glob.glob(SOURCE_PATH)]

    for count, image in enumerate(images):
        result = detect_bolts(image, net)
        cv2.imwrite(RESULT_PATH.format(num = count+1), result)
    print('Finished.')