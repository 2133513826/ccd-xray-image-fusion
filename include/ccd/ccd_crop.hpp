#pragma once

#include "ccd_sequence.hpp"
#include "io/data_loader.hpp"

#include<opencv2/core.hpp>
#include<vector>
#include<string>


//一个物体的信息以及裁剪图
struct CcdObjectCrop{
    CcdObjectInfo object_info;
    
    cv::Rect crop_box;

    //裁剪图
    cv::Mat depth;
    cv::Mat intensity;
    cv::Mat mask;

};

std::vector<CcdObjectCrop> cropCcdObjects(
    const CcdPair& pair,
    const CcdFrameAnalysis& frame,
    const cv::Mat& mask,
    int padding_pixels
);




