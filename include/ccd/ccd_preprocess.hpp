#pragma once
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>

#include"io/data_loader.hpp"
//ccd处理参数
struct CcdOptions{
    double invalid_depth;
    double valid_ratio;
    double depth_diff;
    double intensity_diff;
    int kernel;
    int min_area;
    int edge_margin;
    int padding_pixels;
};
//ccd背景图像
struct CcdBackground{
    cv::Mat depth;
    cv::Mat intensity;
    cv::Mat field;
};

//根据多帧图像建立背景
CcdBackground buildCcdBackground(const std::vector<CcdPair>& CcdPairs,
    const CcdOptions& p);

//比较当前图片与背景，返回目标掩膜
cv::Mat detectCcd(const cv::Mat& depth,const cv::Mat& intensity,
    const CcdBackground& b,const CcdOptions& p);