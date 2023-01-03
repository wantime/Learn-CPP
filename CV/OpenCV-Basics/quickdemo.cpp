//
// Created by wangyao on 2022/12/31.
//

#include "quickopencv.h"
#include <opencv2/dnn.hpp>

void QuickDemo::colorSpace_Demo(cv::Mat &image) {
    Mat gray, hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("HSV", hsv);
    imshow("GRAY", gray);

    imwrite("F:\\hsv.jpg", hsv);
    imwrite("F:\\gray.jpg", gray);
}

void QuickDemo::mat_creation_Demo(Mat &image) {

    Mat m1, m2;
    // 两种深拷贝方式
    m1 = image.clone();
    image.copyTo(m2);

    // 创建空白图像
    Mat m3 = Mat::zeros(Size(514, 514), CV_8UC3);
    m3 = Scalar(127, 227, 27);
    std::cout << "width:" << m3.cols << " height:" << m3.rows << " channels:" << m3.channels() << std::endl;
    //std::cout << m3 << std::endl;
    imshow("m3_1", m3);

    // 直接赋值的浅拷贝
    Mat m4 = m3;
    m4 = Scalar(0, 255, 255);
    imshow("m3_2", m3);
}

void QuickDemo::pixel_visit_Demo(Mat &image) {
    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            // 单通道 灰度图像
            if (dims == 1) {
                int pv = image.at<uchar>(row, col);
                image.at<uchar>(row, col) = 255 - pv;
            }
            // 3通道 彩色图像
            if (dims == 3) {
                Vec3b bgr = image.at<Vec3b>(row, col);
                image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
                image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
                image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
            }
        }
    }
    imshow("inverted color", image);

    for (int row = 0; row < h; row++) {
        uchar *current_row = image.ptr<uchar>(row);
        for (int col = 0; col < w; col++) {
            // 单通道 灰度图像
            if (dims == 1) {
                int pv = *current_row;
                *current_row++ = 255 - pv;
            }
            // 3通道 彩色图像
            if (dims == 3) {
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
            }
        }
    }
    imshow("inverted color 2", image);
}

void QuickDemo::operators_Demo(Mat &image) {
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    m = Scalar(2, 2, 2);

    dst = image + Scalar(50, 50, 50);
    imshow("add", dst);

    dst = image - Scalar(50, 50, 50);
    imshow("subtract", dst);

    dst = image / Scalar(50, 50, 50);
    imshow("divide by", dst);

    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            // 单通道 灰度图像
            if (dims == 1) {
                int p1 = image.at<uchar>(row, col);
                int p2 = m.at<uchar>(row, col);
                // 算术运算
                dst.at<uchar>(row, col) = p1 * p2;
            }
            // 3通道 彩色图像
            if (dims == 3) {
                Vec3b p1 = image.at<Vec3b>(row, col);
                Vec3b p2 = m.at<Vec3b>(row, col);
                // 算术运算
                dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] - p2[0]);
                dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] - p2[1]);
                dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] - p2[2]);
            }
        }
    }
    imshow("multiply", dst);

    // 直接调用接口
    add(image, m, dst);
    subtract(image, m, dst);
    divide(image, m, dst);
    multiply(image, m, dst);

}


static void on_lightness(int b, void *userdata) {
    Mat image = *((Mat *) userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    addWeighted(image, 1.0, m, 0, b, dst);
    imshow("Light&Contrast Modify", dst);
}


static void on_contrast(int b, void *userdata) {
    Mat image = *((Mat *) userdata);
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());
    double contrast = b / 100.0;

    addWeighted(image, contrast, m, 0.0, 0, dst);
    imshow("Light&Contrast Modify", dst);
}

void QuickDemo::tracking_bar_Demo(Mat &image) {
    namedWindow("Light&Contrast Modify", WINDOW_AUTOSIZE);
    int max_value = 100;
    int lightness = 50;
    int contrast_value = 100;
    createTrackbar("Value Bar", "Light&Contrast Modify", &lightness, max_value, on_lightness, (void *) (&image));
    createTrackbar("Contrast Bar", "Light&Contrast Modify", &contrast_value, 200, on_contrast, (void *) (&image));
    on_lightness(50, &image);

}

void QuickDemo::key_Demo(Mat &image) {
    Mat dst = Mat::zeros(image.size(), image.type());;
    while (true) {
        int c = waitKey(100);
        if (c == 27) {
            break;
        }
        if (c == 49) {// key #1
            std::cout << "you enter key #1" << std::endl;
            cvtColor(image, dst, COLOR_BGR2GRAY);
        }
        if (c == 50) {// key #2
            std::cout << "you enter key #2" << std::endl;
            cvtColor(image, dst, COLOR_BGR2HSV);
        }
        if (c == 51) {// key #3
            std::cout << "you enter key #3" << std::endl;
            dst = Scalar(50, 50, 50);
            add(image, dst, dst);
        }
        imshow("key response", dst);
    }
}

void QuickDemo::color_style_Demo(Mat &image) {
    int colormap[] = {
            COLORMAP_AUTUMN,
            COLORMAP_BONE,
            COLORMAP_JET,
            COLORMAP_WINTER,
            COLORMAP_RAINBOW,
            COLORMAP_OCEAN,
            COLORMAP_SUMMER,
            COLORMAP_SPRING,
            COLORMAP_COOL,
            COLORMAP_PINK,
            COLORMAP_HOT,
            COLORMAP_PARULA,
            COLORMAP_MAGMA,
            COLORMAP_INFERNO,
            COLORMAP_PARULA,
            COLORMAP_VIRIDIS,
            COLORMAP_CIVIDIS,
            COLORMAP_TWILIGHT,
            COLORMAP_TWILIGHT_SHIFTED
    };
    Mat dst;
    int index = 0;
    while (true) {
        int c = waitKey(400);
        if (c == 27) {
            break;
        }
        applyColorMap(image, dst, colormap[index]);
        index = (++index) % 19;
        imshow("colormap", dst);
    }
}

void QuickDemo::bitwise_Demo(Mat &image) {
    Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
    Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
    rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 0, 255), -1, LINE_8, 0);
    rectangle(m2, Rect(140, 140, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);

    imshow("m1", m1);
    imshow("m2", m2);

    Mat dst;
    bitwise_and(m1, m2, dst);
    imshow("and", dst);

    bitwise_or(m1, m2, dst);
    imshow("or", dst);

    bitwise_xor(m1, m2, dst);
    imshow("xor", dst);

    //bitwise_not(image, dst);
    //imshow("not", dst);
}

void QuickDemo::channels_Demo(Mat &image) {
    std::vector<Mat> mv;
    split(image, mv);
    imshow("b", mv[0]);
    imshow("g", mv[1]);
    imshow("r", mv[2]);

    Mat dst;
    mv[1] = 0;
    // mv[2] = 0;
    merge(mv, dst);
    imshow("noR", dst);

    int from_to[] = {0, 2, 1, 1, 2, 0};
    mixChannels(&image, 1, &dst, 1, from_to, 3);
    imshow("mix channel", dst);
}

void QuickDemo::inrange_Demo(Mat &image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
    imshow("Mash", mask);

    Mat redbg = Mat::zeros(image.size(), image.type());
    redbg = Scalar(40, 40, 100);
    bitwise_not(mask, mask);
    image.copyTo(redbg, mask);
    imshow("ROI", redbg);
}


void QuickDemo::pixel_statistic_Demo(cv::Mat &image) {
    double minv, maxv;
    Point minLoc, maxLoc;

    std::vector<Mat> mv;
    split(image, mv);
    for (int i = 0; i < image.channels(); i++) {
        minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());
        std::cout << "min value:" << minv << " Max value:" << maxv << std::endl;
    }
    Mat mean, stddev;
    meanStdDev(image, mean, stddev);

    std::cout << "means:" << mean << std::endl;
    std::cout << "stddev:" << stddev << std::endl;

}

void QuickDemo::draw_Demo(Mat &image) {
    Rect rect;

    int x1 = 400;
    int y1 = 200;
    int w1 = 200;
    int h1 = 250;
    rect.x = x1;
    rect.y = y1;
    rect.width = w1;
    rect.height = h1;
    Mat bg = Mat::zeros(image.size(), image.type());
    // 矩形
    rectangle(bg, rect, Scalar(0, 0, 255), -1, 8, 0);
    // 圆形
    circle(bg, Point(x1, y1), 15, Scalar(255, 0, 0), -1, 8, 0);
    circle(bg, Point(x1 + w1, y1), 15, Scalar(255, 0, 0), -1, 8, 0);
    circle(bg, Point(x1, y1 + h1), 15, Scalar(255, 0, 0), -1, 8, 0);
    circle(bg, Point(x1 + w1, y1 + h1), 15, Scalar(255, 0, 0), -1, 8, 0);
    // 线
    line(bg, Point(x1, y1), Point(x1 + w1, y1 + h1), Scalar(0, 255, 0), 2, 8, 0);
    //椭圆
    RotatedRect rrt;
    rrt.center = Point((2 * x1 + w1) / 2, (2 * y1 + h1) / 2);
    rrt.size = Size(w1 / 2, h1 / 2);
    rrt.angle = 90.0;
    ellipse(bg, rrt, Scalar(0, 255, 255), -1, 8);

    // 添加类似透明模板的效果
    Mat dst;
    addWeighted(image, 0.7, bg, 0.3, 0, dst);
    imshow("rect", dst);

}

void QuickDemo::random_drawing_Demo(Mat &image) {
    Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
    int w = canvas.cols;
    int h = canvas.rows;

    RNG rng(12345);
    while (true) {
        int c = waitKey(50);
        if (c == 27) {//退出
            break;
        }
        int x1 = rng.uniform(0, w);
        int y1 = rng.uniform(0, h);
        int x2 = rng.uniform(0, w);
        int y2 = rng.uniform(0, h);

        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        //canvas = Scalar(0,0,0);
        line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 1, LINE_AA);

        imshow("canvas", canvas);
    }
}

void QuickDemo::poly_drawing_Demo(Mat &image) {
    Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
    Point p1(100, 100);
    Point p2(350, 100);
    Point p3(450, 280);
    Point p4(320, 450);
    Point p5(80, 400);
    std::vector<Point> pts;
    pts.push_back(p1);
    pts.push_back(p2);
    pts.push_back(p3);
    pts.push_back(p4);
    pts.push_back(p5);

    // 单个多边形绘制
    //polylines(canvas, pts, true, Scalar(0, 0, 255),52 ,8,0);
    //fillPoly(canvas, pts, Scalar(255, 255, 0), LINE_AA, 0);
    //多个多边形绘制
    std::vector<std::vector<Point>> contours;
    drawContours(canvas, contours, -1, Scalar(255, 0, 0), 2);


    imshow("poly region", canvas);

}

Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;

static void on_draw_rectangle(int event, int x, int y, int flags, void *userdata) {
    Mat image = *((Mat *) (userdata));
    if (event == EVENT_LBUTTONDOWN) {
        sp.x = x;
        sp.y = y;
        std::cout << "start point:" << sp << std::endl;
    } else if (event == EVENT_LBUTTONUP) {
        ep.x = x;
        ep.y = y;
        int dx = ep.x - sp.x;
        int dy = ep.y - sp.y;
        if (dx > 0 && dy > 0) {
            Rect rect(sp.x, sp.y, dx, dy);
            temp.copyTo(image);
            // 获取ROI 需要进一步判断rect的范围是否有效
            imshow("ROI", image(rect));

            rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
            imshow("mouse draw", image);

            // ready for next drawing
            sp.x = -1;
            sp.y = -1;
        }
    } else if (event == EVENT_MOUSEMOVE) {
        if (sp.x > 0 && sp.y > 0) {
            ep.x = x;
            ep.y = y;
            int dx = ep.x - sp.x;
            int dy = ep.y - sp.y;
            if (dx > 0 && dy > 0) {
                Rect rect(sp.x, sp.y, dx, dy);
                temp.copyTo(image);
                rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
                imshow("mouse draw", image);
            }
        }
    }
}

static void on_draw_circle(int event, int x, int y, int flags, void *userdata) {
    Mat image = *((Mat *) (userdata));
    if (event == EVENT_LBUTTONDOWN) {
        sp.x = x;
        sp.y = y;
        std::cout << "start point:" << sp << std::endl;
    } else if (event == EVENT_LBUTTONUP) {
        ep.x = x;
        ep.y = y;
        int dx = ep.x - sp.x;
        int dy = ep.y - sp.y;
        if (dx > 0 && dy > 0) {
            Rect rect(sp.x, sp.y, dx, dy);
            //Circle
            Point center(ep.x, ep.y);
            int r = sqrt(dx * dx + dy * dy);
            circle(image, center, r, Scalar(255, 0, 0), 2, 8, 0);

            imshow("mouse draw", image);
            // 获取ROI
            imshow("ROI", image(rect));
            // ready for next drawing
            sp.x = -1;
            sp.y = -1;
        }
    } else if (event == EVENT_MOUSEMOVE) {
        if (sp.x > 0 && sp.y > 0) {
            ep.x = x;
            ep.y = y;
            int dx = ep.x - sp.x;
            int dy = ep.y - sp.y;
            if (dx > 0 && dy > 0) {
                Rect rect(sp.x, sp.y, dx, dy);
                Point center(ep.x, ep.y);
                int r = sqrt(dx * dx + dy * dy);

                temp.copyTo(image);
                circle(image, center, r, Scalar(255, 0, 0), 2, 8, 0);
                imshow("mouse draw", image);
            }
        }
    }
}


void QuickDemo::mouse_drawing_Demo(Mat &image) {
    namedWindow("mouse draw", WINDOW_AUTOSIZE);
    setMouseCallback("mouse draw", on_draw_rectangle, (void *) (&image));
    imshow("mouse draw", image);
    temp = image.clone();
}

void QuickDemo::norm_Demo(Mat &image) {
    Mat dst;
    std::cout << image.type() << std::endl;
    // CV_8UC3 -> CV_32FC3
    image.convertTo(image, CV_32F);
    std::cout << image.type() << std::endl;
    normalize(image, dst, 1.0, 0, NORM_MINMAX);
    std::cout << dst.type() << std::endl;
    imshow("float data", image);
}

void QuickDemo::resize_Demo(Mat &image) {
    Mat zoomin, zoomout;
    int h = image.rows;
    int w = image.cols;
    resize(image, zoomout, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
    imshow("zoomout", zoomout);
    resize(image, zoomin, Size(w * 2, h * 2), 0, 0, INTER_LINEAR);
    imshow("zoomin", zoomin);


}

void QuickDemo::flip_Demo(Mat &image) {
    Mat dst;
    // 上下翻转
    flip(image, dst, 0);
    // 左右翻转
    flip(image, dst, 1);
    // 对角线,180旋转
    flip(image, dst, -1);
    imshow("flip", dst);
}

void QuickDemo::rotate_Demo(Mat &image) {
    Mat dst, M;
    int w = image.cols;
    int h = image.rows;
    M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);
    double cos_theta = abs(M.at<double>(0, 0));
    double sin_theta = abs(M.at<double>(0, 1));
    int new_w = cos_theta * w + sin_theta * h;
    int new_h = cos_theta * h + sin_theta * w;
    M.at<double>(0, 2) = M.at<double>(0, 2) + (new_w / 2 - w / 2);
    M.at<double>(1, 2) = M.at<double>(1, 2) + (new_h / 2 - h / 2);
    warpAffine(image, dst, M, Size(new_w, new_h), INTER_LINEAR, 0, Scalar(255, 255, 0));
    imshow("rotation", dst);
}

void QuickDemo::video_Demo(Mat &image) {
    VideoCapture capture(0);
    Mat frame;

    int colormap[] = {
            COLORMAP_AUTUMN,
            COLORMAP_BONE,
            COLORMAP_JET,
            COLORMAP_WINTER,
            COLORMAP_RAINBOW,
            COLORMAP_OCEAN,
            COLORMAP_SUMMER,
            COLORMAP_SPRING,
            COLORMAP_COOL,
            COLORMAP_PINK,
            COLORMAP_HOT,
            COLORMAP_PARULA,
            COLORMAP_MAGMA,
            COLORMAP_INFERNO,
            COLORMAP_PARULA,
            COLORMAP_VIRIDIS,
            COLORMAP_CIVIDIS,
            COLORMAP_TWILIGHT,
            COLORMAP_TWILIGHT_SHIFTED
    };
    int index = 0;
    while (true) {
        capture.read(frame);
        if (frame.empty()) {
            break;
        }

//        index = c;
//        applyColorMap(frame, dst, colormap[index]);
        flip(frame, frame, 1);
        imshow("frame", frame);
        // Todo: do something;
        int c = waitKey(10);
        if (c == 27) {
            break;
        }
    }
    capture.release();
}

void QuickDemo::video_process_Demo(Mat &image) {
    VideoCapture capture(".mp4");
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);
    std::cout << "frame width:" << frame_width << std::endl;
    std::cout << "frame height:" << frame_height << std::endl;
    std::cout << "FPS:" << fps << std::endl;
    std::cout << "Number of Frames:" << count << std::endl;
    // color_map_trans
    VideoWriter writer("save.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);
    Mat frame;
    capture.read(frame);

    writer.write(frame);
    writer.release();
    capture.release();
}

void QuickDemo::histogram_Demo(Mat &image) {
    // 分离三个通道
    std::vector<Mat> bgr_plane;
    split(image, bgr_plane);
    // 参数变量
    const int channels[1] = {0};
    const int bins[1] = {256};
    float hranges[2] = {0, 255};
    const float *ranges[1] = {hranges};
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    // 计算Blue, Green, Red通道的直方图
    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
    // 显示直方图
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / bins[0]);
    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);

    // 归一化直方图数据
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // 绘制直方图曲线
    for (int i = 1; i < bins[0]; i++) {
        line(histImage,
             Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
        line(histImage,
             Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0), 2, 8, 0);
        line(histImage,
             Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }
    namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
    imshow("Histogram Demo", histImage);


}

void QuickDemo::histogram_2d_Demo(Mat &image) {
    Mat hsv, hs_hist;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    imshow("hsv", hsv);
    int hbins = 30, sbins = 32;
    int hist_bins[] = {hbins, sbins};
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    const float *hs_ranges[] = {h_range, s_range};
    int hs_channels[] = {0, 1};
    calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
    double maxVal = 0;
    minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
    int scale = 10;
    Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            float binVal = hs_hist.at<float>(h, s);
            // 计算intensity
            int intensity = cvRound(binVal * 255 / maxVal);
            // 对应位置绘制矩形
            rectangle(hist2d_image, Point(h * scale, s * scale),
                      Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                      Scalar::all(intensity),
                      -1);
        }
    }
    applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
    imshow("H-S Histogram", hist2d_image);
    //imwrite("hist_2d.png", hist2d_image);
}

void QuickDemo::histogram_eq_Demo(Mat &image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("gray", gray);
    Mat dst;
    equalizeHist(gray, dst);
    imshow("histogram equalize", dst);
}

void QuickDemo::blur_Demo(Mat &image) {
    Mat dst;
    GaussianBlur(image, dst, Size(0, 0), 15);
    imshow("Gaussian Blur", dst);
    bilateralFilter(image, dst, 0, 100, 10);
    imshow("bilateral filter", dst);
}

void QuickDemo::face_detection_Demo() {
    std::string root_dir = "F:\\download\\opencv\\sources\\samples\\dnn\\";
    dnn::Net net = dnn::readNetFromTensorflow(root_dir + "opencv_face_detector_uint8.pb",
                                              root_dir + "opencv_face_detector.pbtxt");
    VideoCapture capture("F:/download/LoveActually.mp4");
    Mat frame;
    while (true) {
        capture.read(frame);
        if (frame.empty()) {
            break;
        }
        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123),
                                      false, false);
        // NCHW
        net.setInput(blob);
        Mat probs = net.forward();
        Mat detectionMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
        //解析结果
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.5) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                Rect box(x1, y1, x2-x1, y2-y1);
                rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
        imshow("face detection", frame);
        int c = waitKey(1);
        if(c == 27){
            break;
        }
    }

}


