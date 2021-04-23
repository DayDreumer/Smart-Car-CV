#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

Mat calculateLightPattern(Mat img)
{
    Mat pattern;
    //通过使用相对于图像大小的大内核尺寸模糊得到背景图
    blur(img, pattern, Size(img.cols / 3, img.cols / 3));
    return pattern;
}

Mat removeLight(Mat img, Mat pattern)
{
    Mat result;
    Mat img32, pattern32;
    img.convertTo(img32, CV_32F);
    pattern.convertTo(pattern32, CV_32F);
    //通过背景图像移除背景
    result = 1 - (pattern32 / img32);
    result = result * 255;
    result.convertTo(result, CV_8U);
    /*result = img-pattern;*/
    return result;
}

Mat preprocessImage(Mat input) {
    Mat result;
    Mat image_denoise;
    GaussianBlur(input, image_denoise, Size(5,5), 0.8, 0); //高斯滤波处理
    Mat image_no_light;
    Mat light_pattern = calculateLightPattern(image_denoise);
    image_no_light = removeLight(image_denoise, light_pattern);
    threshold(image_no_light, result, 10, 255, THRESH_BINARY);

    return result;
}

struct num_contours {
    double x, y;
}num_contours[300];

int main() {
    cout << "OpenCV_Version: " << CV_VERSION << endl;
    //读取图片
    Mat img_a = imread("D:/visual studio 2019 code list/image_process/num/img.jpg"); //源图像
    //Mat img_after = preprocessImage(img_a);  //预处理
    Mat img,img_b;

    //预处理
    Mat blur_img, usm;
    GaussianBlur(img_a, blur_img, Size(0, 0), 25);
    addWeighted(img_a, 1.5, blur_img, -0.5, 0, img_a);
    

    Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0); //卷积核
    filter2D(img_a, img_a, img.depth(), kernel); //图像卷积运算，提高对比度
    GaussianBlur(img, img, Size(5,5), 3, 3); //高斯滤波 （输入mat，输出mat，内核大小， sigmaX, sigmaY）
    
    cvtColor(img_a,img,CV_BGR2GRAY); //变灰度图
    threshold(img, img, 140, 255, THRESH_BINARY); //变二值图
    
    img.copyTo(img_b);
    vector<vector<Point>> contours; //检测到的轮廓
    vector<Vec4i> hierarchy; //输出向量
    findContours(img_b, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE); // 找轮廓
    Mat resultImage = Mat::zeros(img_b.size(), CV_16UC1); 
    drawContours(resultImage, contours, -1, Scalar(255, 0, 255)); //将轮廓画到resultImage
    
    //通过轮廓提取最小矩形
    vector<Rect> rect; 
    threshold(resultImage, resultImage, 128, 255, THRESH_BINARY_INV);
    Rect min_bounding_rect, re_rect;  //建立两个矩形对象
    float r_x, r_y, width, height;
    for (int i = 0; i < contours.size(); i++) {
        if(contourArea(contours[i])>50){
        min_bounding_rect = boundingRect(contours[i]); //找到包含第i个轮廓的最小矩形，存储到min_bounding_rect
        rect.push_back(boundingRect(contours[i])); //存入rect

        r_x = (float)min_bounding_rect.tl().x; //左上角的x坐标
        r_y = (float)min_bounding_rect.br().y; //左上角的y坐标
        width = (float)min_bounding_rect.width;
        height = (float)min_bounding_rect.height;

        num_contours[i].x = (r_x * 2 + width) / 2.0;
        num_contours[i].y = (r_x * 2 + height) / 2.0;
        }
    }
    
    //ROI处理
    vector<Mat> roi_rect(contours.size());
    for (int j = 0; j < rect.size(); j++) {
        //int j = 156;
        img_a(rect[j]).copyTo(roi_rect[j]);
        cvtColor(roi_rect[j], roi_rect[j], COLOR_BGR2GRAY);
        //threshold(roi_rect[j], roi_rect[j], 140, 255, THRESH_BINARY);
        
    }

    
    //显示
    namedWindow("picture", WINDOW_NORMAL); //新建窗口，WINDOW_NORMAL允许用户自由拖拽窗口大小，WINDOW_AUTOLIZE自适应窗口大小，且不可手动更改
    imshow("picture", roi_rect[3]);
    //imshow("picture", img_after);
    waitKey(0);
    
    return 0;
}