#pragma once
#include "io/data_loader.hpp"

struct XrayObject {
    std::int64_t timestamp_ms = 0; // 被裁切帧的真实拍摄时间。
    cv::Rect bounding_rect;        // 目标外接矩形，采用来源原图坐标。
    std::size_t object_id = 0;     // 全部输出物体中的编号，从1开始。
    cv::Mat high;                  // 裁切后的X光图像。
    cv::Mat mask;                  // 裁切后、只属于该物体的mask。

    // 实际裁切范围，包含padding，采用来源原图坐标。
    cv::Rect crop_box;
};

std::vector<XrayObject> processXrayObjects(
    const XrayData& data,
    const XrayOptions& options
);