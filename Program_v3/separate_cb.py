# 棋盘格尺寸: 5 个参数, 从左到右依次为 宽度、高度、左侧宽度、右侧宽度、上方高度 (尺寸为方格数量)
CHESSBOARD_SIZE = (19, 19, 5, 5, 5)
# CHESSBOARD_SIZE=(21,19,5,5,5)
# 需要识别的图片的路径
INPUT_PATH = './0831图片/08.jpg'


import cv2
from numba import jit
import numpy as np
import time
import math
from random import randrange
from copy import deepcopy


class Graph:
    def __init__(self, num: int):
        self.num = num
        self.neighbour = []
        for i in range(0, num):
            self.neighbour.append([])

    def findAll(self, thisGroup: list, last: int, visited: list):
        thisGroup.append(last)
        visited[last] = True
        for i in self.neighbour[last]:
            if (not visited[i]):
                thisGroup = self.findAll(thisGroup, i, visited)
        return thisGroup

    def add(self, s1: int, s2: int):
        self.neighbour[s1].append(s2)
        self.neighbour[s2].append(s1)

    def getGroups(self):
        visited = [False] * self.num
        groups = []
        for i in range(0, self.num):
            if (not visited[i]):
                thisGroup = []
                groups.append(self.findAll(thisGroup, i, visited))
        return groups


relationship = None


def getCanny(filePath: str, para1: int = 200, para2: int = 400):
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, para1, para2)
    return canny


@jit(nopython=True)
def addAll(l: np.ndarray):
    sum = np.int16(0)
    for i in l:
        for j in i:
            sum += j
    return sum


@jit(nopython=True)
def resize(canny: np.ndarray):
    hight = len(canny)
    width = len(canny[0])
    newHight = hight // 4
    newWidth = width // 4
    new = np.zeros((newHight, newWidth), np.bool8)
    for i in range(0, newHight):
        for j in range(0, newWidth):
            area = canny[4 * i:4 * i + 4, 4 * j:4 * j + 4]
            sum = addAll(area)
            if (sum >= 1):
                new[i][j] = True
    return new


@jit(nopython=True)
def getAllReachableArea2(boolImg: np.ndarray, x: int, y: int, area: np.ndarray):
    height = len(boolImg)
    width = len(boolImg[0])
    allPoints = np.empty((height * width, 2), np.int16)
    pointNum = 0
    if (boolImg[x][y]):
        return allPoints[0:pointNum]
    area[x][y] = True
    subNewPoints = np.empty((10000, 2), np.int16)  # 前一次新增出的点
    subNewPoints[0][0] = x
    subNewPoints[0][1] = y
    allPoints[0][0] = x
    allPoints[0][1] = y
    pointNum = 1
    subNewPointNum = 1
    newPoints = np.empty((40000, 2), np.int16)
    newPointNum = 0
    while True:
        for i in range(0, subNewPointNum):
            x = subNewPoints[i][0]
            y = subNewPoints[i][1]
            newPoints[newPointNum][0] = x - 1
            newPoints[newPointNum][1] = y
            newPoints[newPointNum + 1][0] = x + 1
            newPoints[newPointNum + 1][1] = y
            newPoints[newPointNum + 2][0] = x
            newPoints[newPointNum + 2][1] = y - 1
            newPoints[newPointNum + 3][0] = x
            newPoints[newPointNum + 3][1] = y + 1
            newPointNum += 4
        subNewPointNum = 0
        for i in range(0, newPointNum):
            x = newPoints[i][0]
            y = newPoints[i][1]
            if ((not boolImg[x][y]) and (not area[x][y])):
                area[x][y] = True
                subNewPoints[subNewPointNum][0] = x
                subNewPoints[subNewPointNum][1] = y
                allPoints[pointNum][0] = x
                allPoints[pointNum][1] = y
                subNewPointNum += 1
                pointNum += 1
        newPointNum = 0
        if (subNewPointNum == 0):
            return allPoints[0:pointNum]


@jit(nopython=True)
def boolsToInt(bools: np.ndarray):
    height = len(bools)
    width = len(bools[0])
    img = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            if (bools[i][j]):
                img[i][j] = 255
    return img


def writeBoolImg(boolImg: np.ndarray, filePath: str):
    img = boolsToInt(boolImg)
    cv2.imwrite(filePath, img)


@jit(nopython=True)
def addWhiteEdge(boolImg: np.ndarray):
    height = len(boolImg)
    width = len(boolImg[0])
    newImg = np.ones((height, width), np.bool8)
    newImg[1:-1, 1:-1] = boolImg[1:-1, 1:-1]
    return newImg


@jit(nopython=True)
def getCorner(points: np.ndarray, Type):
    # type  1: 左上, 2: 右上, 3: 左下, 4: 右下
    pointNum = len(points)
    mode1 = (Type % 2) * 2 - 1
    mode2 = ((4 - Type) // 2) * 2 - 1
    sum = np.empty((pointNum,), np.int16)
    num = 0
    for p in points:
        sum[num] = p[0] * mode2 + p[1] * mode1
        num += 1
    m = sum.argmin()
    return points[m]


@jit(nopython=True)
def angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray):
    # 第一个向量 point1 -> point2
    # 第二个向量 point1 -> point3
    # 返回两个向量的夹角
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x3 = point3[0]
    y3 = point3[1]
    product = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1)
    l1 = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    l2 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
    if (l1 == 0.0 or l2 == 0.0):
        return 0
    cos = product / (l1 * l2)
    pi = math.pi
    return math.acos(cos) / pi * 180


@jit(nopython=True)
def distance(point: np.ndarray, line_point1: np.ndarray, line_point2: np.ndarray):
    # 点到直线距离, 直线由两个点确定
    x1 = point[0]
    y1 = point[1]
    x2 = line_point1[0]
    y2 = line_point1[1]
    x3 = line_point2[0]
    y3 = line_point2[1]
    l = ((x3 - x2)**2 + (y3 - y2)**2)**0.5
    if (l == 0.0):
        return 10000.0
    area2 = x1 * y2 + x2 * y3 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3
    return abs(area2 / l)


@jit(nopython=True)
def center(upleft: np.ndarray, upright: np.ndarray, downleft: np.ndarray, downright: np.ndarray):
    cen = np.zeros((2,), np.int16)
    if (upleft[0] - downright[0] == 0 or upright[0] - downleft[0] == 0):
        return cen
    k1 = (upleft[1] - downright[1]) / (upleft[0] - downright[0])
    k2 = (upright[1] - downleft[1]) / (upright[0] - downleft[0])
    b1 = upleft[1] - k1 * upleft[0]
    b2 = upright[1] - k2 * upright[0]
    if (k1 == k2):
        return cen
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    cen[0] = round(x)
    cen[1] = round(y)
    return cen


@jit(nopython=True)
def isLegalSquare(area: np.ndarray, points: np.ndarray, pointMin: int, pointMax: int):
    result = np.zeros((6, 2), np.int16)
    pointNum = len(points)
    if (pointNum < pointMin):
        return result
    if (pointMax < pointNum):
        return result
    result[0][1] = pointNum
    upleft = getCorner(points, 1)
    upright = getCorner(points, 2)
    downleft = getCorner(points, 3)
    downright = getCorner(points, 4)
    result[2] = upleft
    result[3] = upright
    result[4] = downleft
    result[5] = downright
    angle1 = angle(upleft, upright, downleft)  # 左上
    angle2 = angle(upright, upleft, downright)  # 右上
    angle3 = angle(downleft, upleft, downright)  # 左下
    angle4 = angle(downright, upright, downleft)  # 右下
    if (min(angle1, angle2, angle3, angle4) <= 60.0):  # 暂定 60 度
        return result
    if (max(angle1, angle2, angle3, angle4) >= 120.0):  # 暂定 120 度
        return result
    x1 = upleft[0]
    y1 = upleft[1]
    x2 = upright[0]
    y2 = upright[1]
    x3 = downleft[0]
    y3 = downleft[1]
    x4 = downright[0]
    y4 = downright[1]
    l1 = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    l2 = ((x3 - x1)**2 + (y3 - y1)**2)**0.5
    l3 = ((x2 - x4)**2 + (y2 - y4)**2)**0.5
    l4 = ((x3 - x4)**2 + (y3 - y4)**2)**0.5
    m = min(l1, l2, l3, l4)
    if (m == 0.0):
        return result
    if (max(l1, l2, l3, l4) / m > 1.5):  # 暂定 1.5 倍
        return result
    rightVec = np.zeros((2,), np.int16)
    rightVec[1] = 10
    downVec = np.zeros((2,), np.int16)
    downVec[0] = 10
    angle5 = angle(upleft, upright, upleft + rightVec)
    angle6 = angle(downleft, downright, downleft + rightVec)
    angle7 = angle(upleft, downleft, upleft + downVec)
    angle8 = angle(upright, downright, upright + downVec)
    if (max(angle5, angle6, angle7, angle8) > 30.0):  # 暂定 30 度
        return result
    edges = np.empty((pointNum, 2), np.int16)
    edgeNum = 0
    for i in range(0, pointNum):
        x = points[i][0]
        y = points[i][1]
        if (not (area[x - 1][y] and area[x + 1][y] and area[x][y - 1] and area[x][y + 1])):
            edges[edgeNum][0] = x
            edges[edgeNum][1] = y
            edgeNum += 1
    edges = edges[0:edgeNum]
    badPoint = 0
    lines = np.empty((4, 2, 2), np.int16)
    lines[0][0] = upleft
    lines[0][1] = upright
    lines[1][0] = downleft
    lines[1][1] = downright
    lines[2][0] = upleft
    lines[2][1] = downleft
    lines[3][0] = upright
    lines[3][1] = downright
    for e in edges:
        d1 = distance(e, upleft, upright)
        d2 = distance(e, upleft, downleft)
        d3 = distance(e, upright, downright)
        d4 = distance(e, downleft, downright)
        if (min(d1, d2, d3, d4) > 3.0):
            if (not isInside(lines, e[0], e[1])):
                return result
            badPoint += 1
    if (badPoint > 8):
        return result
    cent = center(upleft, upright, downleft, downright)
    if (cent[0] == 0 and cent[1] == 0):
        return result
    result[1] = cent
    result[0][0] = 1
    return result


@jit(nopython=True)
def addWhiteArea(area1: np.ndarray, area2: np.ndarray):
    for p in area2:
        area1[p[0]][p[1]] = True


@jit(nopython=True)
def findAllSquares(boolImg: np.ndarray):
    squares = np.empty((1000, 6, 2), np.int16)
    squareNum = 0
    height = len(boolImg)
    width = len(boolImg[0])
    blackList = boolImg.copy()
    times = 0
    # pointMin=round(width*height*0.0002)
    # pointMax=round(width*height*0.005)
    pointMin = round(width * height * 0.0001)
    pointMax = round(width * height * 0.002)
    area = np.zeros((height, width), np.bool8)
    while True:
        h = randrange(0, height)
        w = randrange(0, width)
        if (blackList[h][w]):
            times += 1
            if (times >= 100000):
                break
            continue
        points = getAllReachableArea2(boolImg, h, w, area)
        addWhiteArea(blackList, points)
        thisSquare = isLegalSquare(area, points, pointMin, pointMax)
        if (thisSquare[0][0] == 0):
            continue
        squares[squareNum] = thisSquare
        squareNum += 1
    return squares[0:squareNum]


@jit(nopython=True)
def showAllSquares(squares, height, width):
    img = np.zeros((height, width), np.bool8)
    for s in squares:
        img[s[1][0]][s[1][1]] = True
        img[s[2][0]][s[2][1]] = True
        img[s[3][0]][s[3][1]] = True
        img[s[4][0]][s[4][1]] = True
        img[s[5][0]][s[5][1]] = True
    return img


@jit(nopython=True)
def getAllCorners(squares: np.ndarray):
    num = len(squares)
    corners = np.empty((num * 4, 2), np.int16)
    for i in range(0, num):
        corners[4 * i:4 * i + 4] = squares[i][2:]
    return corners


@jit(nopython=True)
def getCornerNeighbourNum(corners: np.ndarray, square: np.ndarray):
    result = np.zeros((4,), np.int16) - np.ones((4,), np.int16)
    for i in range(0, 4):
        this = square[i + 2]
        for c in corners:
            if (((this[0] - c[0])**2 + (this[1] - c[1])**2)**0.5 < 4.9):
                result[i] += 1
    return result


@jit(nopython=True)
def getLegalSquares(squares: np.ndarray):
    num = len(squares)
    legalSquares = np.empty((num, 6, 2), np.int16)
    legalNum = 0
    corners = getAllCorners(squares)
    for i in range(0, num):
        nums = getCornerNeighbourNum(corners, squares[i])
        nums = 1000 * nums[0] + 100 * nums[1] + 10 * nums[2] + nums[3]
        # 数量需根据棋盘格调整
        if (nums == 3333 or nums == 1223 or nums == 2132 or nums == 2312 or nums == 3221 or nums == 2233 or nums == 3232 or nums == 3322 or nums == 2323):
            legalSquares[legalNum] = squares[i]
            legalNum += 1
    return legalSquares[0:legalNum]


@jit(nopython=True)
def isTogether(square1: np.ndarray, square2: np.ndarray):
    for c1 in square1[2:]:
        for c2 in square2[2:]:
            if (((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5 < 6):
                return True
    return False


@jit(nopython=True)
def getRelationshipTable(squares: np.ndarray):
    num = len(squares)
    table = np.zeros((num, num), np.bool8)
    for i in range(0, num):
        for j in range(0, num):
            if (isTogether(squares[i], squares[j])):
                table[i][j] = True
    return table


def formGroup(squares: np.ndarray):
    num = len(squares)
    table = getRelationshipTable(squares)
    g = Graph(num)
    for i in range(0, num):
        for j in range(0, num):
            if (j >= i):
                continue
            if (table[i][j]):
                g.add(i, j)
    groups = g.getGroups()
    cnt = 0
    for group in groups:
        if (len(group) >= 20):
            cnt += 1
    if (cnt == 0):
        return [False, '没有数量超过20的正方形组']
    if (cnt >= 2):
        return [False, '发现一个以上数量超过20的正方形组']
    for group in groups:
        if (len(group) >= 20):
            group.sort()
            return [True, np.array(group, np.int16)]
    return groups


def findAllTogetherSquares(squares: np.ndarray, indexes: np.ndarray, legalSquares: np.ndarray):
    num = len(squares)
    ind = indexes[0]
    s = legalSquares[ind]
    for k in range(0, num):
        if (np.array_equal(s, squares[k])):
            ind = k
            break
    table = getRelationshipTable(squares)
    g = Graph(num)
    for i in range(0, num):
        for j in range(0, num):
            if (j >= i):
                continue
            if (table[i][j]):
                g.add(i, j)
    groups = g.getGroups()
    for group in groups:
        if (ind in group):
            group.sort()
            n = len(group)
            finalSquares = np.empty((n, 6, 2), np.int16)
            for i in range(0, n):
                finalSquares[i] = squares[group[i]]
            return finalSquares


@jit(nopython=True)
def getRelationship(square1: np.ndarray, square2: np.ndarray):
    # 返回值: 上 1  下 2  左 3  右 4  左上 5  右上 6  左下 7  右下 8
    if (np.array_equal(square2, square1)):
        return -1
    r = (
        (-1, 3, 1, 5),
        (4, -1, 6, 1),
        (2, 7, -1, 3),
        (8, 2, 4, -1)
    )
    for i in range(0, 4):
        c1 = square1[i + 2]
        for j in range(0, 4):
            c2 = square2[j + 2]
            if (((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5 < 6):
                return r[i][j]
    return -1


@jit(nopython=True)
def getAllRelationship(squares: np.ndarray):
    num = len(squares)
    table = np.zeros((num, 8), np.int16) - np.ones((num, 8), np.int16)
    for i in range(0, num):
        for j in range(0, num):
            rela = getRelationship(squares[i], squares[j])
            if (rela > 0):
                table[i][rela - 1] = j
    return table


def findNeighbours(chessboard: list, this: int, location: list, visited: list):
    visited.append(this)
    chessboard[location[0]][location[1]] = this
    #  上 0  下 1  左 2  右 3  左上 4  右上 5  左下 6  右下 7
    directions = (
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    )
    for i in range(0, 8):
        if (relationship[this][i] == -1 or (relationship[this][i] in visited)):
            continue
        dire = directions[i]
        findNeighbours(chessboard, relationship[this][i], [location[0] + dire[0], location[1] + dire[1]], visited)


@jit(nopython=True)
def removeEdges(cb: np.ndarray):
    upMax = 49
    downMax = 0
    leftMax = 49
    rightMax = 0
    for i in range(0, 50):
        for j in range(0, 50):
            if (cb[i][j] > -1):
                upMax = min(upMax, i)
                downMax = max(downMax, i)
                leftMax = min(leftMax, j)
                rightMax = max(rightMax, j)
    return cb[upMax:downMax + 1, leftMax:rightMax + 1]


def drawTable(squares: np.ndarray):
    global relationship
    num = len(squares)
    relationship = getAllRelationship(squares)
    cb = []
    visited = []
    for i in range(0, 50):
        cb.append([-1] * 50)
    findNeighbours(cb, 0, [25, 25], visited)
    return removeEdges(np.array(cb, np.int16))


def check(cb: np.ndarray):
    height = len(cb)
    width = len(cb[0])
    if (height != CHESSBOARD_SIZE[1] or width != CHESSBOARD_SIZE[0]):
        return [False, '尺寸不符']
    minmax = [0, 0]
    results = []
    # 左边
    for i in range(0, height):
        if (cb[i][0] > -1):
            minmax[0] = i
            break
    for i in range(0, height):
        if (cb[height - i - 1][0] > -1):
            minmax[1] = height - i - 1
            break
    if (minmax[1] - minmax[0] < 4):
        return [False, '左侧格点缺失']
    results.append(deepcopy(minmax))
    # 右边
    for i in range(0, height):
        if (cb[i][-1] > -1):
            minmax[0] = i
            break
    for i in range(0, height):
        if (cb[height - i - 1][-1] > -1):
            minmax[1] = height - i - 1
            break
    if (minmax[1] - minmax[0] < 4):
        return [False, '右侧格点缺失']
    results.append(deepcopy(minmax))
    # 上方
    for i in range(0, width):
        if (cb[0][i] > -1):
            minmax[0] = i
            break
    for i in range(0, width):
        if (cb[0][width - i - 1] > -1):
            minmax[1] = width - i - 1
            break
    if (minmax[1] - minmax[0] < 4):
        return [False, '上方格点缺失']
    results.append(deepcopy(minmax))
    # 下方
    for i in range(0, width):
        if (cb[-1][i] > -1):
            minmax[0] = i
            break
    for i in range(0, width):
        if (cb[-1][width - i - 1] > -1):
            minmax[1] = width - i - 1
            break
    if (minmax[1] - minmax[0] < 4):
        return [False, '下方格点缺失']
    results.append(deepcopy(minmax))
    # 左侧最右列
    l = CHESSBOARD_SIZE[2] - 1
    for i in range(0, height):
        if (cb[i][l] > -1):
            minmax[0] = i
            break
    for i in range(0, height):
        if (cb[height - i - 1][l] > -1):
            minmax[1] = height - i - 1
            break
    if (minmax[1] - minmax[0] < 5):
        return [False, '左侧最右列格点缺失']
    results.append(deepcopy(minmax))
    # 右侧最左列
    l = -CHESSBOARD_SIZE[3] + width
    for i in range(0, height):
        if (cb[i][l] > -1):
            minmax[0] = i
            break
    for i in range(0, height):
        if (cb[height - i - 1][l] > -1):
            minmax[1] = height - i - 1
            break
    if (minmax[1] - minmax[0] < 5):
        return [False, '右侧最左列格点缺失']
    results.append(deepcopy(minmax))
    # 上侧最下行
    l = CHESSBOARD_SIZE[4] - 1
    for i in range(0, width):
        if (cb[l][i] > -1):
            minmax[0] = i
            break
    for i in range(0, width):
        if (cb[l][width - i - 1] > -1):
            minmax[1] = width - i - 1
            break
    if (minmax[1] - minmax[0] < 6):
        return [False, '上侧最下行格点缺失']
    results.append(deepcopy(minmax))
    return [True, results]


def avgSize(squares):
    n = len(squares)
    sum = 0
    for s in squares:
        sum += s[0][1]
    return sum / n


def getSeparateLines(squares: np.ndarray, corners: list, cb: np.ndarray):
    l = (avgSize(squares)**0.5) * 1
    leftup = squares[cb[corners[0][0]][0]][1]
    leftdown = squares[cb[corners[0][1]][0]][1]
    rightup = squares[cb[corners[1][0]][-1]][1]
    rightdown = squares[cb[corners[1][1]][-1]][1]
    subleftup = squares[cb[corners[5][0]][-CHESSBOARD_SIZE[3] + CHESSBOARD_SIZE[0]]][1]  # 右侧最左列
    subleftdown = squares[cb[corners[5][1]][-CHESSBOARD_SIZE[3] + CHESSBOARD_SIZE[0]]][1]
    subrightup = squares[cb[corners[4][0]][CHESSBOARD_SIZE[2] - 1]][1]  # 左侧最右列
    subrightdown = squares[cb[corners[4][1]][CHESSBOARD_SIZE[2] - 1]][1]
    upleft = squares[cb[0][corners[2][0]]][1]
    upright = squares[cb[0][corners[2][1]]][1]
    downleft = squares[cb[-1][corners[3][0]]][1]
    downright = squares[cb[-1][corners[3][1]]][1]
    subdownleft = squares[cb[CHESSBOARD_SIZE[4] - 1][corners[6][0]]][1]  # 上侧最下行
    subdownright = squares[cb[CHESSBOARD_SIZE[4] - 1][corners[6][1]]][1]
    result = [
        [[4 * (upleft[0] - l) + 2, upleft[1] * 4 + 2], [4 * (upright[0] - l) + 2, upright[1] * 4 + 2]],  # 0. 上方
        [[4 * (downleft[0] + l) + 2, downleft[1] * 4 + 2], [4 * (downright[0] + l) + 2, downright[1] * 4 + 2]],  # 1. 下方
        [[leftup[0] * 4 + 2, 4 * (leftup[1] - l) + 2], [leftdown[0] * 4 + 2, 4 * (leftdown[1] - l) + 2]],  # 2. 左边
        [[rightup[0] * 4 + 2, 4 * (rightup[1] + l) + 2], [rightdown[0] * 4 + 2, 4 * (rightdown[1] + l) + 2]],  # 3. 右边
        [[subrightup[0] * 4 + 2, 4 * (subrightup[1] + l / 4) + 2], [subrightdown[0] * 4 + 2, 4 * (subrightdown[1] + l / 4) + 2]],  # 4. 左侧最右列
        [[subleftup[0] * 4 + 2, 4 * (subleftup[1] - l / 4) + 2], [subleftdown[0] * 4 + 2, 4 * (subleftdown[1] - l / 4) + 2]],  # 5. 右侧最左列
        [[4 * (subdownleft[0] + l / 4) + 2, subdownleft[1] * 4 + 2], [4 * (subdownright[0] + l / 4) + 2, subdownright[1] * 4 + 2]],  # 6. 上侧最下行 (往下平移)
        [[4 * (subdownleft[0] - l / 4) + 2, subdownleft[1] * 4 + 2], [4 * (subdownright[0] - l / 4) + 2, subdownright[1] * 4 + 2]]  # 7. 上侧最下行 (往上平移)
    ]
    return np.array(result, np.float64)


@jit(nopython=True)
def isInside(lines: np.ndarray, x: int, y: int):
    x1 = lines[0][0][0]
    y1 = lines[0][0][1]
    x2 = lines[0][1][0]
    y2 = lines[0][1][1]
    x3 = lines[1][0][0]
    y3 = lines[1][0][1]
    x4 = lines[1][1][0]
    y4 = lines[1][1][1]
    s = ((x1 - x) * (y2 - y) - (y1 - y) * (x2 - x)) * ((x3 - x) * (y4 - y) - (y3 - y) * (x4 - x))
    if (s > 0):
        return False
    x1 = lines[2][0][0]
    y1 = lines[2][0][1]
    x2 = lines[2][1][0]
    y2 = lines[2][1][1]
    x3 = lines[3][0][0]
    y3 = lines[3][0][1]
    x4 = lines[3][1][0]
    y4 = lines[3][1][1]
    s = ((x1 - x) * (y2 - y) - (y1 - y) * (x2 - x)) * ((x3 - x) * (y4 - y) - (y3 - y) * (x4 - x))
    if (s > 0):
        return False
    return True


@jit(nopython=True)
def getPartChessboard(lines: np.ndarray, gray: np.ndarray):
    gray = gray.copy()
    height = len(gray)
    width = len(gray[0])
    for i in range(0, height):
        for j in range(0, width):
            if (not isInside(lines, i, j)):
                gray[i][j] = 255
    return gray


@jit(nopython=True)
def getPartChessboard2(lines: np.ndarray, gray: np.ndarray):
    gray = gray.copy()
    height = len(gray)
    width = len(gray[0])
    for i in range(0, height):
        for j in range(0, width):
            if (isInside(lines, i, j)):
                gray[i][j] = 255
    return gray


def init():
    global CHESSBOARD_SIZE
    backup = CHESSBOARD_SIZE
    CHESSBOARD_SIZE = (12, 9, 5, 4, 4)
    t = time.time()
    canny = getCanny('./compile_test.jpg')
    resized = resize(canny)
    img = boolsToInt(resized)
    we = addWhiteEdge(resized)
    square = findAllSquares(we)
    ls = getLegalSquares(square)
    g = formGroup(ls)
    img = showAllSquares(square, len(resized), len(resized[0]))
    finalSquares = findAllTogetherSquares(square, g[1], ls)
    cb = drawTable(finalSquares)
    corners = check(cb)[1]
    lines = getSeparateLines(square, corners, cb)
    gray = cv2.cvtColor(cv2.imread('./compile_test.jpg'), cv2.COLOR_BGR2GRAY)
    up = getPartChessboard(lines[[0, 6, 2, 3]], gray)
    t = time.time() - t
    print('编译成功，花费 ' + str(t) + ' 秒')
    CHESSBOARD_SIZE = backup


def separateChessboard2(filePath=None):
    if (filePath == None):
        filePath = INPUT_PATH
    t = time.time()
    canny = getCanny(filePath, 200, 400)
    # print(time.time() - t)
    resized = resize(canny)
    we = addWhiteEdge(resized)
    square = findAllSquares(we)
    if (len(square) == 0):
        return [False, '未发现正方形']
    ls = getLegalSquares(square)
    g = formGroup(ls)
    if (g[0] == False):
        return [False, g[1]]
    finalSquares = findAllTogetherSquares(square, g[1], ls)
    cb = drawTable(finalSquares)
    corners = check(cb)
    if (not corners[0]):
        return [False, corners[1]]
    corners = corners[1]
    lines = getSeparateLines(finalSquares, corners, cb)
    # print(time.time() - t)
    gray = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2GRAY)
    up = getPartChessboard2(lines[[6, 1, 2, 3]], gray)
    coverUp = getPartChessboard2(lines[[0, 7, 2, 3]], gray)
    left = getPartChessboard2(lines[[7, 1, 4, 3]], coverUp)
    right = getPartChessboard2(lines[[7, 1, 2, 5]], coverUp)
    imgs = np.empty((3, len(up), len(up[0])), np.uint8)
    imgs[0] = up
    imgs[1] = left
    imgs[2] = right
    # cv2.imwrite('./chessboard_up.jpg', up)
    # cv2.imwrite('./chessboard_left.jpg', left)
    # cv2.imwrite('./chessboard_right.jpg', right)
    # print(time.time() - t)
    return [True, imgs]


def getCorner2(points: np.ndarray, Type: str):
    # type  upleft, upright, downleft, downright
    dic = {
        'upleft': [1, 1],
        'upright': [-1, 1],
        'downleft': [1, -1],
        'downright': [-1, -1]
    }
    mode = dic[Type]
    sum = []
    for p in points:
        sum.append(p[0][0] * mode[0] + p[0][1] * mode[1])
    loc = sum.index(min(sum))
    return points[loc][0]


def getFourCorners(filePath=None):
    r = separateChessboard2(filePath)
    if (not r[0]):
        return [False, r[1]]
    up = r[1][0]
    left = r[1][1]
    right = r[1][2]
    try:
        ret, upCorners = cv2.findChessboardCorners(up, (CHESSBOARD_SIZE[0]-1, CHESSBOARD_SIZE[4]-1), None)
        upCorners = cv2.cornerSubPix(up, upCorners, (5, 4), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        # print(len(upCorners))
        ret, leftCorners = cv2.findChessboardCorners(left, (CHESSBOARD_SIZE[2]-1, CHESSBOARD_SIZE[1]-CHESSBOARD_SIZE[4]), None)
        leftCorners = cv2.cornerSubPix(left, leftCorners, (5, 4), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        # print(len(leftCorners))
        ret, rightCorners = cv2.findChessboardCorners(right, (CHESSBOARD_SIZE[3]-1, CHESSBOARD_SIZE[1]-CHESSBOARD_SIZE[4]), None)
        rightCorners = cv2.cornerSubPix(right, rightCorners, (5, 4), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        # print(len(rightCorners))
    except:
        return [False, '棋盘格角点未识别到']
    result = np.zeros((4, 2), np.float64)
    result[0] = getCorner2(upCorners, 'upleft')
    result[1] = getCorner2(upCorners, 'upright')
    result[2] = getCorner2(leftCorners, 'downleft')
    result[3] = getCorner2(rightCorners, 'downright')
    return [True, result]


if (__name__ == '__main__'):
    init()
    print(getFourCorners())
