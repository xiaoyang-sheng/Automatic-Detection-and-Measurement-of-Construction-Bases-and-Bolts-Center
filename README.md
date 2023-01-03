# SJTU_IPP_Program2022-Automatic-detection-and-measurement-of-construction-bases-and-bolts-center

This is the **working repository** of the project for the 24th University Innovation and Practice Program of Shanghai Jiao Tong University in 2022. **This is not the final repo/version.** 
The detailed information could be checked in the file of "研究论文", which is the project paper in Chinese.

## Key tools and knowledge:
Computer Vision, Construction bases/bolts detection, OpenCV, Holistically-Nested Edge Detection, Special-shaped calibration tools
## Abstract
This program uses Computer Vision technology to develop an automatic detecting and measuring tool based on Python and OpenCV, which could be used for the circular bases on the construction sites. It can detect and measure the distance difference between the center of outside circular base and the center of bolts on it. Therefore, the tool can identify potential constructing errors and avoid safety problems. Besides, the program can greatly save the human labors, instead of manually measuring the distance. The program uses some basic knowledge of camera calibration, perspective transformation, together with Holistically-Nested Edge Detection (HED), Hough Circle Detection, Canny Edge Detection, KMeans clustering and other self-developed classification and detection methods. We also designed a special pi-shape chessboard for calibration to match the structure of the bases, and implement corresponding algorithms. With respect to a circular base of 1.5m in diameter and a 40cm rectangle made by the 4 foundation bolts, the error of the detection of the corresponding centers are all under 3mm, which is good enough for measuring usage.

The website of the project in Shanghai Jiao Tong University IPP program: 
<https://cxcy.sjtu.edu.cn/CXCY/SJTU/Item/Detail/e50a0e0f-1033-450d-aa2f-a8adb00848f9>

## 中文项目名称：同组地脚螺栓中心对主柱中心偏移的自动智能检查

这是2022第二十四期上海交通大学大学生创新创业训练计划项目“同组地脚螺栓中心对主柱中心偏移的自动智能检查”的工作库，**并不是最终完整整理版本。** 具体详细内容可以参考“研究论文”文档。

## 上海交通大学大学生创新创业训练计划项目该项目立项详细网站：
<https://cxcy.sjtu.edu.cn/CXCY/SJTU/Item/Detail/e50a0e0f-1033-450d-aa2f-a8adb00848f9>

## Group

Professor: Sung-Liang Chen (陈松良)
<https://umji.sjtu.edu.cn/~slchen/>

Memember (contributors):

- Zeyu Zhang
- Xiaoyang Sheng 
- Tiancheng Jiang
- Zhaoyuan Liang
- Zhengwei Wang
