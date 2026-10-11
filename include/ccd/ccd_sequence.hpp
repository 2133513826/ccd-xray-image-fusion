#pragma once

#include <opencv2/opencv.hpp>
#include<vector>
#include<string>
#include<cstddef>
#include<cstdint>
//物体姿态
enum class CcdObjectPosition{
    Interior,
    TopBoundary,
    BottomBoundary
};
//当前帧的一个连通区域信息
struct CcdObjectInfo{
    std::size_t file_index;//文件编号
    std::int64_t timestamp_ms=0;//时间戳
    int label=0;//连通区域编号
    int area;//连通区域面积
    cv::Rect bounding_rect;//连通区域外接矩形   
    cv::Point2d center;//连通区域质心
    CcdObjectPosition position;//物体位置
};




//单帧信息与所有连通区域信息
struct CcdFrameAnalysis
{
    std::size_t file_index; //文件编号
    std::int64_t timestamp_ms=0;//时间戳
    std::vector<CcdObjectInfo> objects;//连通区域信息
    cv::Mat labels;//连通区域标签
};

//分析一帧ccd的信息
CcdFrameAnalysis analyzeCcdMask(
    std::size_t file_index,
    const std::int64_t timestamp_ms,
    const cv::Mat& mask,
    const int edge_margin);