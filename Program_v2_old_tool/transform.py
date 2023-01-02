import cv2
import numpy as np

def OnMouseEvent( event, x, y, flags, param):
    global lbtDownPos
    global pos
    global pointList
    img = param
    ignoreEvent = [cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONUP, cv2.EVENT_MBUTTONDBLCLK, cv2.EVENT_MOUSEWHEEL,
                        cv2.EVENT_MOUSEHWHEEL,cv2.EVENT_MOUSEMOVE,cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP]  # 需要忽略的鼠标事件
    needRecordEvent = [ cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP]  # 需要记录当前信息的鼠标事件

    if event == cv2.EVENT_LBUTTONUP:
        pos = (x,y)
        print("OnMouseEvent EVENT_LBUTTONUP:",pos)
        n = len(pointList)
        if pos==lbtDownPos:



            n += 1
            if n <= 3:
                pointList.append(pos)
                cv2.putText(img, '.', (x - 10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 0))
                cv2.putText(img, f'select point{n}:({x},{y})', (x + 20, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0))

        lbtDownPos = None

    elif event == cv2.EVENT_LBUTTONDOWN:
        lbtDownPos = (x,y)
        print("OnMouseEvent EVENT_LBUTTONDOWN:", lbtDownPos)

    else:lbtDownPos = None


def getPoint(imgfile):
    global pos
    global pointList

    pointList = []
    img = cv2.imread(imgfile)
    cv2.putText(img, 'https://blog.csdn.net/LaoYuanPython', (100, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(255, 0, 0))
    imgbak = np.array(img)
    rows,cols = img.shape[:2]
    winName = 'select three point'
    cv2.namedWindow(winName, 0)
    cv2.resizeWindow(winName, 500, 500)  # 初始窗口大小
    cv2.setMouseCallback(winName, OnMouseEvent, img)
    print("请将要单独放大的部分从其左上角、左下角、右下角分别鼠标左键点击选择三个点，选择后在图像上有提示信息，选择完成后按ESC退出")


    while True:#通过鼠标左键点击选择三个点，分别代表要映射到左上、左下和右下三个点
        cv2.imshow(winName, img)
        ch = cv2.waitKey(100)
        if ch == 27: break

    destPoint = [(0,0),(0,rows),(cols,rows)]
    if len(pointList)==3:
        pts1 = np.float32(pointList)
        pts2 = np.float32(destPoint)
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(imgbak, M, (cols, rows))
        cv2.namedWindow('demo', 0)  # 0 窗口可伸缩
        cv2.resizeWindow('demo', 500, 500)  # 初始窗口大小
        cv2.imshow("demo", dst)  # 展示图片
        ch = cv2.waitKey(0)
    else:
        print("没有选择足够的点")




getPoint(r'C:\Users\dell\Desktop\Courses\IPP\serverV2\calibresult4.jpg')
