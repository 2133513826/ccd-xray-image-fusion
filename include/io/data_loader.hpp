#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <vector>
#include <filesystem>
#include<cstdint>
#include <regex>

struct XrayFrame {
    std::filesystem::path path;
    std::int64_t timestamp_ms=0;
    cv::Mat high, mask; 
    int labels=0;
};

bool loadCcdPair(const std::string&depth_path,
                  const std::string&intensity_path,
                  cv::Mat& depth_image,
                  cv::Mat& intensity_image);

struct CcdFilePair{
    std::string id;
    std::string depth_path;
    std::string intensity_path;
};

struct CcdPair{
    std::string id;
    cv::Mat depth_image;
    cv::Mat intensity_image;
};

std::vector<CcdFilePair> inspectCcdFolder(const std::string& folder);

std::vector<CcdPair> loadCcdPairs(const std::vector<CcdFilePair>& file_pairs);


struct XrayOptions {
    int frame_interval_ms=185, time_tolerance_ms=5;
    int foreground_threshold=0, morph_kernel=3, min_area=300;
    int edge_margin=3, padding_pixels=5;
    int valid_x_begin=70, valid_x_end=910; 
    // 跨帧物体关联时，横向区间IoU的最低要求。
double min_horizontal_iou = 0.7;

// 确认物体亮核心的高阈值，使用原始16位灰度。
int core_threshold = 5000;

// 每个候选连通域至少包含多少个高于上述阈值的像素。
int min_core_pixels = 20;
};
struct XrayData {
    std::vector<std::vector<XrayFrame>> groups;
};
cv::Mat generateXrayMask(
    const cv::Mat& high,
    const XrayOptions& options
);
XrayData loadXrayData(
    const std::filesystem::path& high_dir,
    const XrayOptions& options
);