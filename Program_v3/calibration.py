# 棋盘格宽度和高度
WIDTH = 18
HEIGHT = 18
# 照片的路径
PATH = "./imgs/亚克力板_标定"
# 照片数量
IMG_NUM = 16
# xlsx 文件保存位置
OUTPUT = './相机参数.xlsx'

import cv2
import numpy as np
from openpyxl import Workbook


# 获取每张照片的路径
def getPath(n: int):
    n = str(n)
    if (len(n) == 1):
        n = '0' + n
    return f'{PATH}/{n}.jpg'


# 初始化
corners = []
objPoint = np.zeros((WIDTH * HEIGHT, 3), np.float32)
objPoint[:, :2] = np.mgrid[0:WIDTH, 0:HEIGHT].T.reshape(-1, 2)
objPoints = []

# 棋盘格角点识别
for i in range(1, IMG_NUM + 1):
    print(f"第{i}张照片开始识别")
    img = cv2.imread(getPath(i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(gray, (WIDTH, HEIGHT), None)
    # corner 有可能返回 None, 但不能直接用 ==None 判断
    try:
        if (len(corner) != WIDTH * HEIGHT):
            print(f"第{i}张照片未成功识别, 请检查原因")
            continue
        else:
            print(f"第{i}张照片识别成功")
    except:
        print(f"第{i}张照片未成功识别, 请检查原因")
        continue
    corner = cv2.cornerSubPix(gray, corner, (5, 4), (-1, -1),
                              (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
    corners.append(corner)
    objPoints.append(objPoint)

# 相机标定
size_all = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objPoints, corners, size_all, None, None)

# 相机参数优化
dist = dist[0:4]
h, w = gray.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print(mtx, dist, newcameramtx)


# 以下为 Excel 部分
ALPHAS = '.ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUMS = {}
for i in range(1, 27):
    NUMS[ALPHAS[i]] = i


def numToAlpha(n):
    s = ''
    while (n > 0):
        t = n % 26
        if (t == 0):
            t = 26
        s = ALPHAS[t] + s
        n = (n - t) // 26
    return s


def alphaToNum(a):
    l = len(a)
    sum = 0
    for i in range(0, l):
        sum += 26**(l - i - 1) * NUMS[a[i]]
    return sum


def cellToTuple(cell: str):
    for i in range(0, len(cell)):
        try:
            a = NUMS[cell[i]]
        except:
            n = i
            break
    return (alphaToNum(cell[:n]), int(cell[n:]))


def tupleToCell(t):
    return numToAlpha(t[0]) + str(t[1])


def write_cell(sheet, i, j, value):
    sheet[tupleToCell((i, j))].value = value


# 写入 Excel 函数
def write_camera_matrix(mtx, dist, newcameramtx):
    wb = Workbook()
    ws = wb.active
    write_cell(ws, 1, 1, float(mtx[0][0]))
    write_cell(ws, 2, 1, float(mtx[0][1]))
    write_cell(ws, 3, 1, float(mtx[0][2]))
    write_cell(ws, 1, 2, float(mtx[1][0]))
    write_cell(ws, 2, 2, float(mtx[1][1]))
    write_cell(ws, 3, 2, float(mtx[1][2]))
    write_cell(ws, 1, 3, float(mtx[2][0]))
    write_cell(ws, 2, 3, float(mtx[2][1]))
    write_cell(ws, 3, 3, float(mtx[2][2]))
    write_cell(ws, 1, 5, float(newcameramtx[0][0]))
    write_cell(ws, 2, 5, float(newcameramtx[0][1]))
    write_cell(ws, 3, 5, float(newcameramtx[0][2]))
    write_cell(ws, 1, 6, float(newcameramtx[1][0]))
    write_cell(ws, 2, 6, float(newcameramtx[1][1]))
    write_cell(ws, 3, 6, float(newcameramtx[1][2]))
    write_cell(ws, 1, 7, float(newcameramtx[2][0]))
    write_cell(ws, 2, 7, float(newcameramtx[2][1]))
    write_cell(ws, 3, 7, float(newcameramtx[2][2]))
    write_cell(ws, 1, 9, float(dist[0][0]))
    write_cell(ws, 2, 9, float(dist[0][1]))
    write_cell(ws, 3, 9, float(dist[0][2]))
    write_cell(ws, 4, 9, float(dist[0][3]))
    write_cell(ws, 5, 9, float(dist[0][4]))
    wb.save(OUTPUT)


# 写入到 Excel
write_camera_matrix(mtx, dist, newcameramtx)
