#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "quickopencv.h"
using namespace cv;
using namespace std;

int main() {
    // 读取图像
    Mat img = imread("lion.jpg", IMREAD_UNCHANGED);
    if(img.empty()){
        cout << "Open Fail" << endl;
        return -1;
    }
     //设置窗口
    namedWindow("Image", WINDOW_FREERATIO);
     //1.展示图片
    imshow("Image", img);
    QuickDemo qd;
    // 2.通道转换
    //qd.colorSpace_Demo(img);
    // 3.创建空白Mat以及深浅拷贝
    //qd.mat_creation_Demo(img);
    // 4.访问像素并反转
    //qd.pixel_visit_Demo(img);
    // 5.算术运算
    //qd.operators_Demo(img);
    // 6.滚动条调整对比度与亮度 addWeight函数 利用track bar传递参数等
    //qd.tracking_bar_Demo(img);
    // 7.按键事件
    //qd.key_Demo(img);
    // 8.颜色表
    //qd.color_style_Demo(img);
    // 9.图像的逻辑运算
    //qd.bitwise_Demo(img);
    // 10.通道
    //qd.channels_Demo(img);
    // 11.图像色彩空间转换与inrange使用
    //qd.inrange_Demo(img);
    // 12.像素值统计
    //qd.pixel_statistic_Demo(img);
    // 13.绘制几何图案
    //qd.draw_Demo(img);
    // 14.随机数与随机颜色
    //qd.random_drawing_Demo(img);
    //imshow("origin", img);
    // 15.多边形绘制与填充
    //qd.poly_drawing_Demo(img);
    // 16.鼠标事件(绘制图形)
    //qd.mouse_drawing_Demo(img);
    // 17.像素值的归一化与类型转换
    //qd.norm_Demo(img);
    // 18. 图像放缩与插值
    //qd.resize_Demo(img);
    // 19. 图像翻转 flip
    // qd.flip_Demo(img);
    // 20. 仿射变换
    //qd.rotate_Demo(img);
    // 21. 摄像头\视频读取
    //qd.video_Demo(img);
    // 22.视频处理与保存
    //qd.video_process_Demo(img);
    // 23.普通直方图
    //qd.histogram_Demo(img);
    // 24.intensity-2D 直方图
    //qd.histogram_2d_Demo(img);
    // 25.直方图均衡
    //qd.histogram_eq_Demo(img);
    // 26.高斯模糊与双边模糊
    //qd.blur_Demo(img);
    // 27.人脸识别demo
    //qd.face_detection_Demo();

    return 0;
}
