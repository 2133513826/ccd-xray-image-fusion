#pragma once

#include "ccd/ccd_crop.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// 连续帧关联参数
struct CcdTemporalOptions {
    std::int64_t max_frame_gap_ms = 450;
    double min_horizontal_overlap = 0.50;

    int min_overlap_rows = 2;
    int max_overlap_rows = 30;

    // 防止少量像素碰巧得到很高IoU
    int min_mask_intersection_pixels = 50;
    int min_valid_depth_pixels = 30;

    // mask形状的最低重合要求
    double min_mask_iou = 0.50;

    double mask_weight = 0.20;
    double depth_weight = 0.50;
    double intensity_weight = 0.30;

    double redundancy_score_threshold = 0.65;

    double invalid_depth = -32768.0;

    double fusion_depth_tolerance = 12.0;//融合深度容差
};

struct CcdOverlapScore {
    int overlap_rows = 0;

    // mask交集和并集的像素数量
    int mask_intersection_pixels = 0;
    int mask_union_pixels = 0;

    // 交集 / 并集
    double mask_iou = 0.0;

    // 深度有效的交集像素数量
    int valid_depth_pixels = 0;
    double mask_coverage = 0.0;

    double depth_score = 0.0;
    double intensity_score = 0.0;
    double combined_score = 0.0;

    bool valid = false;
};

enum class CcdFusionMode {
    DirectConcat,       // 没有冗余，直接上下拼接
    RedundancyFusion    // 存在冗余，需要融合
};

// 融合决策记录
struct CcdFusionDecision {
    // 最终采用的融合方式
    CcdFusionMode mode =
        CcdFusionMode::DirectConcat;

    // 最佳重叠行数
    // DirectConcat时为0
    int best_overlap_rows = 0;

    // 最佳综合得分
    double best_score = 0.0;

    // 保存n=2～30的全部得分
    std::vector<CcdOverlapScore> scores;
};

// 一组可能属于同一物体的跨帧目标
struct CcdCrossFrameMatch {
    // 两个下标都指向输入的 all_objects
    std::size_t previous_index = 0;
    std::size_t current_index = 0;

    std::int64_t time_gap_ms = 0;
    double horizontal_overlap = 0.0;
};

struct CcdObjectMerge {
    CcdCrossFrameMatch match;

    // 保存融合方式以及评分数据
    CcdFusionDecision decision;

    CcdObjectCrop merged_object;
};

//ccd模块最终输出结果
struct CcdTemporalResult {
    //所有结果
    std::vector<CcdObjectCrop> final_ccd_objects;
    // 跨帧匹配结果，用于展示算法结果
    std::vector<CcdObjectMerge> merged_info;
};

// 计算两个外接矩形的横向重叠率
double calculateHorizontalOverlap(
    const cv::Rect& first,
    const cv::Rect& second
);

// 判断两个物体是否满足跨帧候选条件
bool isCrossFrameCandidate(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
);


// 计算指定重叠行数n的评分
CcdOverlapScore evaluateOverlapRows(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    int overlap_rows,
    const CcdTemporalOptions& options
);

// 遍历所有n并决定是否存在冗余
CcdFusionDecision estimateCcdFusion(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
);

// 没有冗余时直接拼接
CcdObjectCrop concatenateCcdObjects(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
);
// 存在冗余时进行处理
CcdObjectCrop fuseCcdRedundantRegion(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    int overlap_rows,
    const CcdTemporalOptions& options
);


// 处理全部物体
CcdTemporalResult processCcdTemporalObjects(
    const std::vector<CcdObjectCrop>& all_objects,
    const CcdTemporalOptions& options
);
