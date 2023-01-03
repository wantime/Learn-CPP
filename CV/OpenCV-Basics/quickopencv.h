//
// Created by wangyao on 2022/12/31.
//

#ifndef LEARN_OPENCV_QUICKOPENCV_H
#define LEARN_OPENCV_QUICKOPENCV_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;

class QuickDemo{
public:
    // 颜色通道转换
    void colorSpace_Demo(Mat &image);
    // 深浅拷贝、创建Mat
    void mat_creation_Demo(Mat &image);
    // 访问像素 索引和指针法
    void pixel_visit_Demo(Mat &image);
    // 像素的算术运算
    void operators_Demo(Mat &image);
    // 滚动条调整参数
    void tracking_bar_Demo(Mat &image);
    // 键盘事件
    void key_Demo(Mat &image);
    // color map
    void color_style_Demo(Mat &image);
    // 与 或 非
    void bitwise_Demo(Mat &image);
    // 通道分离与合并
    void channels_Demo(Mat &image);
    // 图像色彩空间转换
    void inrange_Demo(Mat &image);
    // 像素统计
    void pixel_statistic_Demo(Mat &image);
    // 绘制几何图案
    void draw_Demo(Mat &image);
    // 随机数与随机颜色
    void random_drawing_Demo(Mat &image);
    // 多边形绘制与填充
    void poly_drawing_Demo(Mat &image);
    // 鼠标事件
    void mouse_drawing_Demo(Mat &image);
    // 像素类型转换与归一化操作
    void norm_Demo(Mat &image);
    // 图像缩放和插值
    void resize_Demo(Mat &image);
    // 翻转
    void flip_Demo(Mat &image);
    // 旋转
    void rotate_Demo(Mat &image);
    // 视频读取
    void video_Demo(Mat &image);
    // 视频处理与保存
    void video_process_Demo(Mat &image);
    // 统计直方图
    void histogram_Demo(Mat &image);
    // 2D 统计直方图
    void histogram_2d_Demo(Mat &image);
    // 直方图均衡化
    void histogram_eq_Demo(Mat &image);
    // 模糊(高斯模糊、双边模糊)
    void blur_Demo(Mat &image);
    void face_detection_Demo();
};
#endif //LEARN_OPENCV_QUICKOPENCV_H
