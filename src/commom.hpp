#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include<fstream>
#include<map>
#include<vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinter1;

cv::Rect get_rect(cv::Mat& img,float bbox[4])
{
    
}