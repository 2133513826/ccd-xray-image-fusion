#include "ccd/ccd_temporal.hpp"
#include "common/progress.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <utility>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>


namespace {

// 判断一个深度值是否有效
bool validDepth(
    double value,
    double invalid_depth
) {
    // 深度值不能是NaN、无穷大或者相机规定的无效值
    return std::isfinite(value) &&
           value != invalid_depth;
}

// 计算两组数据的皮尔逊相关性
// 返回值被限制在0～1：
// 1表示高度正相关，0表示无正相关或负相关
double correlationScore(
    const std::vector<double>& first,
    const std::vector<double>& second
) {
    // 两组数据数量必须相同，并且至少需要两个数据
    if (first.size() != second.size() ||
        first.size() < 2) {
        return 0.0;
    }

    // 计算两组数据的平均值
    const double first_mean =
        std::accumulate(
            first.begin(),
            first.end(),
            0.0
        ) / first.size();

    const double second_mean =
        std::accumulate(
            second.begin(),
            second.end(),
            0.0
        ) / second.size();

    double numerator = 0.0;
    double first_square_sum = 0.0;
    double second_square_sum = 0.0;

    // 计算皮尔逊相关系数中的分子和分母部分
    for (std::size_t i = 0;
         i < first.size();
         ++i) {
        const double first_delta =
            first[i] - first_mean;

        const double second_delta =
            second[i] - second_mean;

        numerator +=
            first_delta * second_delta;

        first_square_sum +=
            first_delta * first_delta;

        second_square_sum +=
            second_delta * second_delta;
    }

    const double denominator =
        std::sqrt(
            first_square_sum *
            second_square_sum
        );

    // 如果其中一组数据几乎没有变化，
    // 分母接近0，无法计算可靠的相关性
    if (denominator <= 1e-12) {
        return 0.0;
    }

    const double correlation =
        numerator / denominator;

    // 同一物体应当表现为正相关，
    // 因此负相关直接限制为0
    return std::clamp(
        correlation,
        0.0,
        1.0
    );
}

} // namespace



double calculateHorizontalOverlap(
    const cv::Rect& first,
    const cv::Rect& second
) {
    // 两个矩形交集的左边界
    const int intersection_left =
        std::max(first.x, second.x);

    // 两个矩形交集的右边界
    const int intersection_right =
        std::min(
            first.x + first.width,
            second.x + second.width
        );

    // 如果没有重叠，宽度取0
    const int intersection_width =
        std::max(
            0,
            intersection_right - intersection_left
        );

    // 选择两个矩形中较小的宽度
    const int smaller_width =
        std::min(first.width, second.width);

    // 防止除以0
    if (smaller_width <= 0) {
        return 0.0;
    }

    return static_cast<double>(intersection_width) /
           static_cast<double>(smaller_width);
}


bool isCrossFrameCandidate(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
) {
    const CcdObjectInfo& previous_info =
        previous.object_info;

    const CcdObjectInfo& current_info =
        current.object_info;

    // 1. 前一帧目标必须位于下边界
    if (previous_info.position !=
        CcdObjectPosition::BottomBoundary) {
        return false;
    }

    // 2. 后一帧目标必须位于上边界
    if (current_info.position !=
        CcdObjectPosition::TopBoundary) {
        return false;
    }

    // 3. 两个目标必须来自相邻帧
    if (current_info.file_index <=
        previous_info.file_index) {
        return false;
    }

    if (current_info.file_index -
            previous_info.file_index != 1) {
        return false;
    }

    // 4. 计算时间差
    const std::int64_t time_gap_ms =
        current_info.timestamp_ms -
        previous_info.timestamp_ms;

    if (time_gap_ms <= 0) {
        return false;
    }

    if (time_gap_ms >
        options.max_frame_gap_ms) {
        return false;
    }

    // 5. 计算横向重叠率
    const double horizontal_overlap =
        calculateHorizontalOverlap(
            previous_info.bounding_rect,
            current_info.bounding_rect
        );

    if (horizontal_overlap <
        options.min_horizontal_overlap) {
        return false;
    }

    return true;
}

CcdOverlapScore evaluateOverlapRows(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    int overlap_rows,
    const CcdTemporalOptions& options
) {
    CcdOverlapScore score;

    // 记录当前正在评价的重叠行数n
    score.overlap_rows = overlap_rows;

    // 检查两组裁切图像是否存在
    if (previous.mask.empty() ||
        previous.depth.empty() ||
        previous.intensity.empty() ||
        current.mask.empty() ||
        current.depth.empty() ||
        current.intensity.empty()) {
        throw std::invalid_argument(
            "CCD crop image is empty"
        );
    }

    // n必须大于0，并且不能超过任意一张裁切图的高度
    if (overlap_rows <= 0 ||
        overlap_rows > previous.mask.rows ||
        overlap_rows > current.mask.rows) {
        return score;
    }

    /*
     * previous和current的裁切区域可能具有不同的x坐标和宽度。
     *
     * 因此不能直接使用裁切图中的相同列进行比较，
     * 必须先根据crop_box.x找到它们在原图中的公共横向范围。
     */
    const int global_left =
        std::max(
            previous.crop_box.x,
            current.crop_box.x
        );

    const int global_right =
        std::min(
            previous.crop_box.x +
                previous.mask.cols,
            current.crop_box.x +
                current.mask.cols
        );

    // 横向没有公共区域，当前n无法进行评价
    if (global_left >= global_right) {
        return score;
    }

    /*
     * 将深度图和强度图转换为double。
     * 这样后面计算差值、平均值和相关系数时，
     * 不需要分别处理CV_16S、CV_8U等不同类型。
     */
    cv::Mat previous_depth;
    cv::Mat current_depth;
    cv::Mat previous_intensity;
    cv::Mat current_intensity;

    previous.depth.convertTo(
        previous_depth,
        CV_64F
    );

    current.depth.convertTo(
        current_depth,
        CV_64F
    );

    previous.intensity.convertTo(
        previous_intensity,
        CV_64F
    );

    current.intensity.convertTo(
        current_intensity,
        CV_64F
    );

    // 保存共同目标区域中的深度值
    std::vector<double> previous_depth_values;
    std::vector<double> current_depth_values;

    // 保存共同目标区域中的强度值
    std::vector<double> previous_intensity_values;
    std::vector<double> current_intensity_values;

    /*
     * previous取底部n行：
     *     rows-n ～ rows-1
     *
     * current取顶部n行：
     *     0 ～ n-1
     */
    for (int row = 0;
         row < overlap_rows;
         ++row) {
        const int previous_y =
            previous.mask.rows -
            overlap_rows +
            row;

        const int current_y = row;

        // 只遍历两个裁切区域共同拥有的原图x坐标
        for (int global_x = global_left;
             global_x < global_right;
             ++global_x) {
            /*
             * 把原图坐标转换成两张裁切图各自的局部坐标。
             */
            const int previous_x =
                global_x -
                previous.crop_box.x;

            const int current_x =
                global_x -
                current.crop_box.x;

            const bool previous_foreground =
                previous.mask.at<unsigned char>(
                    previous_y,
                    previous_x
                ) != 0;

            const bool current_foreground =
                current.mask.at<unsigned char>(
                    current_y,
                    current_x
                ) != 0;

            // 至少一张mask属于目标，就是mask并集
            if (previous_foreground ||
                current_foreground) {
                ++score.mask_union_pixels;
            }

            // 只有两张mask都属于目标，才是mask交集
            if (!(previous_foreground &&
                  current_foreground)) {
                continue;
            }

            ++score.mask_intersection_pixels;

            /*
             * 强度图不存在invalid_depth的问题，
             * 因此只要属于mask交集就参与计算。
             */
            previous_intensity_values.push_back(
                previous_intensity.at<double>(
                    previous_y,
                    previous_x
                )
            );

            current_intensity_values.push_back(
                current_intensity.at<double>(
                    current_y,
                    current_x
                )
            );

            const double first_depth =
                previous_depth.at<double>(
                    previous_y,
                    previous_x
                );

            const double second_depth =
                current_depth.at<double>(
                    current_y,
                    current_x
                );

            /*
             * 只有两张图的深度都有效，
             * 这对深度值才参与相关性计算。
             */
            if (validDepth(
                    first_depth,
                    options.invalid_depth) &&
                validDepth(
                    second_depth,
                    options.invalid_depth)) {
                previous_depth_values.push_back(
                    first_depth
                );

                current_depth_values.push_back(
                    second_depth
                );
            }
        }
    }

    // 保存有效深度像素对的数量
    score.valid_depth_pixels =
        static_cast<int>(
            previous_depth_values.size()
        );

    // 计算mask交并比
    if (score.mask_union_pixels > 0) {
        score.mask_iou =
            static_cast<double>(
                score.mask_intersection_pixels
            ) /
            static_cast<double>(
                score.mask_union_pixels
            );
    }

    /*
     * 计算mask覆盖率。
     *
     * 它用于防止n很小时，
     * 少量像素碰巧得到很高IoU和相关性。
     */
    const int previous_mask_area =
        cv::countNonZero(previous.mask);

    const int current_mask_area =
        cv::countNonZero(current.mask);

    const int smaller_mask_area =
        std::min(
            previous_mask_area,
            current_mask_area
        );

    if (smaller_mask_area > 0) {
        score.mask_coverage =
            static_cast<double>(
                score.mask_intersection_pixels
            ) /
            static_cast<double>(
                smaller_mask_area
            );
    }

    // 计算深度和强度的正相关得分
    score.depth_score =
        correlationScore(
            previous_depth_values,
            current_depth_values
        );

    score.intensity_score =
        correlationScore(
            previous_intensity_values,
            current_intensity_values
        );

    // 计算权重总和，防止配置错误
    const double weight_sum =
        options.mask_weight +
        options.depth_weight +
        options.intensity_weight;

    if (weight_sum <= 0.0) {
        throw std::invalid_argument(
            "CCD temporal score weights are invalid"
        );
    }

    /*
     * 先融合三种相似度：
     * mask形状、深度变化和强度变化。
     */
    const double similarity =
        (
            options.mask_weight *
                score.mask_iou +
            options.depth_weight *
                score.depth_score +
            options.intensity_weight *
                score.intensity_score
        ) / weight_sum;

    /*
     * 再乘mask覆盖率。
     *
     * 即使n很小时相关性很高，
     * 因为覆盖的目标区域很少，
     * 最终综合得分仍然不会过高。
     */
    score.combined_score =
        score.mask_coverage *
        similarity;

    /*
     * 当前n必须拥有足够的mask交集像素、
     * 有效深度像素以及足够高的IoU，
     * 才能作为有效候选。
     */
    score.valid =
        score.mask_intersection_pixels >=
            options.min_mask_intersection_pixels &&
        score.valid_depth_pixels >=
            options.min_valid_depth_pixels &&
        score.mask_iou >=
            options.min_mask_iou;

    return score;
}

CcdFusionDecision estimateCcdFusion(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
) {
    CcdFusionDecision decision;

    // n不能超过配置上限，也不能超过任意裁切图高度
    const int maximum_overlap_rows =
        std::min({
            options.max_overlap_rows,
            previous.mask.rows,
            current.mask.rows
        });

    // 从配置中的最小n开始逐个尝试
    for (int overlap_rows =
             options.min_overlap_rows;
         overlap_rows <=
             maximum_overlap_rows;
         ++overlap_rows) {
        CcdOverlapScore score =
            evaluateOverlapRows(
                previous,
                current,
                overlap_rows,
                options
            );

        // 保存每个n的得分，方便后面输出或画曲线
        decision.scores.push_back(score);

        // 无效候选不能参与最佳n选择
        if (!score.valid) {
            continue;
        }

        // 保存目前得分最高的n
        if (score.combined_score >
            decision.best_score) {
            decision.best_score =
                score.combined_score;

            decision.best_overlap_rows =
                overlap_rows;
        }
    }

    /*
     * 最高分达到配置阈值：
     * 认为两帧之间存在冗余区域。
     *
     * 否则认为没有可靠冗余，
     * 后面直接进行上下拼接。
     */
    if (decision.best_score >=
        options.redundancy_score_threshold) {
        decision.mode =
            CcdFusionMode::RedundancyFusion;
    } else {
        decision.mode =
            CcdFusionMode::DirectConcat;
    }

    return decision;
}
CcdObjectCrop concatenateCcdObjects(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    const CcdTemporalOptions& options
) {
    // 1. 检查图像是否存在
    if (previous.depth.empty() ||
        previous.intensity.empty() ||
        previous.mask.empty() ||
        current.depth.empty() ||
        current.intensity.empty() ||
        current.mask.empty()) {
        throw std::invalid_argument(
            "CCD crop image is empty"
        );
    }

    // 每个目标内部的三种图像尺寸必须相同
    if (previous.depth.size() != previous.intensity.size() ||
        previous.depth.size() != previous.mask.size() ||
        current.depth.size() != current.intensity.size() ||
        current.depth.size() != current.mask.size()) {
        throw std::invalid_argument(
            "CCD crop images have different sizes"
        );
    }

    // 两个目标对应图像的数据类型必须相同
    if (previous.depth.type() != current.depth.type() ||
        previous.intensity.type() != current.intensity.type()) {
        throw std::invalid_argument(
            "CCD crop image types are different"
        );
    }

    if (previous.mask.type() != CV_8UC1 ||
        current.mask.type() != CV_8UC1) {
        throw std::invalid_argument(
            "CCD crop mask must have type CV_8UC1"
        );
    }

    /*
     * 2. 计算两张裁切图在原图x方向上的并集。
     *
     * 两张图的crop_box.x可能不同，因此输出画布
     * 不能直接使用其中一张图的宽度。
     */
    const int global_left =
        std::min(
            previous.crop_box.x,
            current.crop_box.x
        );

    const int global_right =
        std::max(
            previous.crop_box.x +
                previous.depth.cols,
            current.crop_box.x +
                current.depth.cols
        );

    const int canvas_width =
        global_right - global_left;

    /*
     * 不存在冗余时，current直接放在previous下方。
     */
    const int current_offset_y =
        previous.depth.rows;

    const int canvas_height =
        previous.depth.rows +
        current.depth.rows;

    if (canvas_width <= 0 ||
        canvas_height <= 0) {
        throw std::runtime_error(
            "Concatenated CCD canvas size is invalid"
        );
    }

    /*
     * 3. 创建输出画布。
     *
     * 深度图没有数据的位置设为invalid_depth；
     * 强度图和mask没有数据的位置设为0。
     */
    cv::Mat merged_depth(
        canvas_height,
        canvas_width,
        previous.depth.type(),
        cv::Scalar(options.invalid_depth)
    );

    cv::Mat merged_intensity =
        cv::Mat::zeros(
            canvas_height,
            canvas_width,
            previous.intensity.type()
        );

    cv::Mat merged_mask =
        cv::Mat::zeros(
            canvas_height,
            canvas_width,
            CV_8UC1
        );

    /*
     * 4. 计算previous在输出画布中的区域。
     */
    const int previous_offset_x =
        previous.crop_box.x -
        global_left;

    const cv::Rect previous_roi(
        previous_offset_x,
        0,
        previous.depth.cols,
        previous.depth.rows
    );

    /*
     * 只复制mask中的目标像素。
     *
     * 裁切图中的背景不会覆盖输出画布，
     * 深度背景继续保持invalid_depth。
     */
    previous.depth.copyTo(
        merged_depth(previous_roi),
        previous.mask
    );

    previous.intensity.copyTo(
        merged_intensity(previous_roi),
        previous.mask
    );

    previous.mask.copyTo(
        merged_mask(previous_roi)
    );

    /*
     * 5. 计算current在输出画布中的区域。
     *
     * current放在previous下方，所以其y偏移为
     * previous.depth.rows。
     */
    const int current_offset_x =
        current.crop_box.x -
        global_left;

    const cv::Rect current_roi(
        current_offset_x,
        current_offset_y,
        current.depth.cols,
        current.depth.rows
    );

    current.depth.copyTo(
        merged_depth(current_roi),
        current.mask
    );

    current.intensity.copyTo(
        merged_intensity(current_roi),
        current.mask
    );

    current.mask.copyTo(
        merged_mask(current_roi)
    );

    /*
     * 6. 构造最终物体。
     */
    CcdObjectCrop result;

    // 使用前一帧作为合并目标的来源信息
    result.object_info =
        previous.object_info;

    // 拼接后认为物体已经完整
    result.object_info.position =
        CcdObjectPosition::Interior;

    /*
     * 合并图已经不再属于某一张原始图片，
     * y坐标从新的拼接画布顶部开始。
     */
    result.crop_box =
        cv::Rect(
            global_left,
            0,
            canvas_width,
            canvas_height
        );

    result.depth =
        std::move(merged_depth);

    result.intensity =
        std::move(merged_intensity);

    result.mask =
        std::move(merged_mask);

    /*
     * 7. 根据最终mask重新计算物体信息。
     */
    result.object_info.area =
        cv::countNonZero(result.mask);

    if (result.object_info.area > 0) {
        // 得到物体在合并图中的局部外接矩形
        const cv::Rect local_box =
            cv::boundingRect(result.mask);

        // x恢复为原CCD图像中的横向坐标
        result.object_info.bounding_rect =
            cv::Rect(
                global_left + local_box.x,
                local_box.y,
                local_box.width,
                local_box.height
            );

        // 重新计算合并物体的质心
        const cv::Moments moments =
            cv::moments(
                result.mask,
                true
            );

        if (moments.m00 > 0.0) {
            result.object_info.center.x =
                global_left +
                moments.m10 / moments.m00;

            result.object_info.center.y =
                moments.m01 / moments.m00;
        }
    }

    return result;
}

CcdTemporalResult processCcdTemporalObjects(
    const std::vector<CcdObjectCrop>& all_objects,
    const CcdTemporalOptions& options
) {
    CcdTemporalResult result;

    // used[i]表示第i个物体是否已经被处理。
    // 跨帧融合后，需要将前后两个物体都标记为已处理，
    // 防止它们再次作为独立物体输出。
    std::vector<bool> used(all_objects.size(), false);

    for (std::size_t i = 0; i < all_objects.size(); ++i) {
        // The first i entries have been visited, including already-used objects.
        showProgress("CCD temporal", i, all_objects.size());
        // 该物体已经参加过跨帧融合，直接跳过。
        if (used[i]) {
            continue;
        }

        const CcdObjectCrop& previous = all_objects[i];
        const CcdObjectInfo& previous_info =
            previous.object_info;

        // 情况一：物体位于图像内部。
        // 可以认为当前帧已经完整拍摄该物体，直接输出。
        if (previous_info.position ==
            CcdObjectPosition::Interior) {
            result.final_ccd_objects.push_back(previous);
            used[i] = true;
            continue;
        }

        // 情况二：物体位于上边界。
        // 按照当前处理规则，它应该已经有机会被前一帧的
        // BottomBoundary物体匹配。
        // 如果运行到这里仍未被使用，说明没有找到前一帧对象，
        // 因此将它作为独立物体保留。
        if (previous_info.position ==
            CcdObjectPosition::TopBoundary) {
            result.final_ccd_objects.push_back(previous);
            used[i] = true;
            continue;
        }

        // 只有下边界物体才需要向后一帧寻找对应目标。
        if (previous_info.position !=
            CcdObjectPosition::BottomBoundary) {
            result.final_ccd_objects.push_back(previous);
            used[i] = true;
            continue;
        }

        // 保存找到的最佳候选物体下标。
        std::size_t best_candidate_index =
            all_objects.size();

        // 当前先使用横向重叠率选择最佳候选。
        double best_horizontal_overlap = -1.0;

        // 在后面的物体中寻找下一帧的上边界物体。
        for (std::size_t j = i + 1;
             j < all_objects.size();
             ++j) {
            if (used[j]) {
                continue;
            }

            const CcdObjectCrop& current = all_objects[j];
            const CcdObjectInfo& current_info =
                current.object_info;

            // 还没有到下一帧，则继续寻找。
            if (current_info.file_index <=
                previous_info.file_index) {
                continue;
            }

            // 已经超过下一帧，后面不可能再出现合适候选。
            if (current_info.file_index >
                previous_info.file_index + 1) {
                break;
            }

            // 统一使用已经写好的候选判断函数检查：
            // 1. 前一物体是否位于下边界；
            // 2. 后一物体是否位于上边界；
            // 3. 是否来自相邻帧；
            // 4. 时间差是否合理；
            // 5. 横向重叠率是否达到要求。
            if (!isCrossFrameCandidate(
                    previous,
                    current,
                    options)) {
                continue;
            }

            const double horizontal_overlap =
                calculateHorizontalOverlap(
                    previous_info.bounding_rect,
                    current_info.bounding_rect
                );

            // 如果下一帧中有多个候选物体，
            // 选择横向重叠程度最高的一个。
            if (horizontal_overlap >
                best_horizontal_overlap) {
                best_horizontal_overlap =
                    horizontal_overlap;
                best_candidate_index = j;
            }
        }

        // 没有找到下一帧候选目标。
        // 当前下边界物体只能作为独立目标保留。
        if (best_candidate_index ==
            all_objects.size()) {
            result.final_ccd_objects.push_back(previous);
            used[i] = true;
            continue;
        }

        const CcdObjectCrop& current =
            all_objects[best_candidate_index];

        // 遍历不同的重叠行数，判断两部分之间是否存在冗余，
        // 同时得到最佳重叠行数和每个n对应的评分。
        CcdFusionDecision decision =
            estimateCcdFusion(
                previous,
                current,
                options
            );

        CcdObjectCrop merged_object;

        if (decision.mode ==
            CcdFusionMode::RedundancyFusion) {
            // 检测到可靠的冗余区域，进行羽化融合。
            merged_object =
                fuseCcdRedundantRegion(
                    previous,
                    current,
                    decision.best_overlap_rows,
                    options
                );
        } else {
            // 没有检测到可靠冗余，直接上下拼接。
            merged_object =
                concatenateCcdObjects(
                    previous,
                    current,
                    options
                );
        }

        // 记录两个原始物体之间的匹配关系。
        CcdCrossFrameMatch match;
        match.previous_index = i;
        match.current_index = best_candidate_index;

        match.time_gap_ms =
            current.object_info.timestamp_ms -
            previous.object_info.timestamp_ms;

        match.horizontal_overlap =
            best_horizontal_overlap;

        // 保存用于展示和评价算法的详细融合记录。
        CcdObjectMerge merge_info;
        merge_info.match = match;
        merge_info.decision = std::move(decision);
        merge_info.merged_object = merged_object;

        // 融合结果进入CCD最终物体集合。
        result.final_ccd_objects.push_back(merged_object);

        // 另存一份融合记录。
        // cv::Mat采用引用计数，所以这里的图像数据是浅拷贝，
        // 不会立刻复制整幅图像。
        result.merged_info.push_back(
            std::move(merge_info)
        );

        // 前后两个原始物体已经合并，后续遍历必须跳过。
        used[i] = true;
        used[best_candidate_index] = true;
    }

    showProgress("CCD temporal", all_objects.size(), all_objects.size());
    return result;
}
CcdObjectCrop fuseCcdRedundantRegion(
    const CcdObjectCrop& previous,
    const CcdObjectCrop& current,
    int overlap_rows,
    const CcdTemporalOptions& options
) {
    // 1. 检查输入图像
    if (previous.depth.empty() ||
        previous.intensity.empty() ||
        previous.mask.empty() ||
        current.depth.empty() ||
        current.intensity.empty() ||
        current.mask.empty()) {
        throw std::invalid_argument(
            "CCD crop image is empty"
        );
    }

    // 每组裁切图中的深度、强度和mask尺寸必须一致
    if (previous.depth.size() != previous.intensity.size() ||
        previous.depth.size() != previous.mask.size() ||
        current.depth.size() != current.intensity.size() ||
        current.depth.size() != current.mask.size()) {
        throw std::invalid_argument(
            "CCD crop images have different sizes"
        );
    }

    if (previous.mask.type() != CV_8UC1 ||
        current.mask.type() != CV_8UC1) {
        throw std::invalid_argument(
            "CCD crop mask must have type CV_8UC1"
        );
    }

    // 两组对应图像的数据类型应当一致
    if (previous.depth.type() != current.depth.type() ||
        previous.intensity.type() != current.intensity.type()) {
        throw std::invalid_argument(
            "CCD crop image types are different"
        );
    }

    // 重叠行数必须在两张图的高度范围内
    if (overlap_rows <= 0 ||
        overlap_rows > previous.depth.rows ||
        overlap_rows > current.depth.rows) {
        throw std::invalid_argument(
            "overlap_rows is outside the valid range"
        );
    }

    /*
     * 2. 计算输出画布的横向范围。
     *
     * previous和current的crop_box.x是它们在原图中的横坐标，
     * 因此使用两者的横向并集建立画布。
     */
    const int global_left =
        std::min(
            previous.crop_box.x,
            current.crop_box.x
        );

    const int global_right =
        std::max(
            previous.crop_box.x +
                previous.depth.cols,
            current.crop_box.x +
                current.depth.cols
        );

    const int canvas_width =
        global_right - global_left;

    /*
     * current顶部与previous底部重叠overlap_rows行。
     *
     * current在画布中的起始y坐标：
     * previous高度 - 重叠行数
     */
    const int current_offset_y =
        previous.depth.rows -
        overlap_rows;

    const int canvas_height =
        previous.depth.rows +
        current.depth.rows -
        overlap_rows;

    if (canvas_width <= 0 ||
        canvas_height <= 0) {
        throw std::runtime_error(
            "Merged CCD canvas size is invalid"
        );
    }

    /*
     * 3. 转换成double进行计算。
     *
     * 原始深度图可能是CV_16S，
     * 强度图可能是CV_8U。
     * 使用double可以安全进行加权运算。
     */
    cv::Mat previous_depth;
    cv::Mat current_depth;
    cv::Mat previous_intensity;
    cv::Mat current_intensity;

    previous.depth.convertTo(
        previous_depth,
        CV_64F
    );

    current.depth.convertTo(
        current_depth,
        CV_64F
    );

    previous.intensity.convertTo(
        previous_intensity,
        CV_64F
    );

    current.intensity.convertTo(
        current_intensity,
        CV_64F
    );

    /*
     * 4. 创建输出图像。
     *
     * 深度图背景初始化为无效深度；
     * 强度图和mask背景初始化为0。
     */
    cv::Mat merged_depth(
        canvas_height,
        canvas_width,
        CV_64F,
        cv::Scalar(options.invalid_depth)
    );

    cv::Mat merged_intensity =
        cv::Mat::zeros(
            canvas_height,
            canvas_width,
            CV_64F
        );

    cv::Mat merged_mask =
        cv::Mat::zeros(
            canvas_height,
            canvas_width,
            CV_8UC1
        );

    /*
     * 5. 遍历输出画布。
     *
     * 对每个画布像素，分别换算它在previous和current
     * 裁切图中的局部坐标。
     */
    for (int canvas_y = 0;
         canvas_y < canvas_height;
         ++canvas_y) {
        for (int canvas_x = 0;
             canvas_x < canvas_width;
             ++canvas_x) {
            const int global_x =
                global_left + canvas_x;

            // previous在画布顶部，因此y坐标不需要偏移
            const int previous_x =
                global_x -
                previous.crop_box.x;

            const int previous_y =
                canvas_y;

            // current向下偏移current_offset_y
            const int current_x =
                global_x -
                current.crop_box.x;

            const int current_y =
                canvas_y -
                current_offset_y;

            // 判断当前坐标是否落在previous图像范围内
            const bool inside_previous =
                previous_x >= 0 &&
                previous_x < previous.mask.cols &&
                previous_y >= 0 &&
                previous_y < previous.mask.rows;

            // 判断当前坐标是否落在current图像范围内
            const bool inside_current =
                current_x >= 0 &&
                current_x < current.mask.cols &&
                current_y >= 0 &&
                current_y < current.mask.rows;

            const bool has_previous =
                inside_previous &&
                previous.mask.at<unsigned char>(
                    previous_y,
                    previous_x
                ) != 0;

            const bool has_current =
                inside_current &&
                current.mask.at<unsigned char>(
                    current_y,
                    current_x
                ) != 0;

            // 两张图中都不是目标，保持背景
            if (!has_previous && !has_current) {
                continue;
            }

            // 最终mask取两张mask的并集
            merged_mask.at<unsigned char>(
                canvas_y,
                canvas_x
            ) = 255;

            double previous_weight = 0.0;
            double current_weight = 0.0;

            if (has_previous && has_current) {
                /*
                 * 当前像素位于冗余区域。
                 *
                 * current_y就是当前像素在重叠区域中的行号：
                 * 0 ～ overlap_rows-1
                 */
                const int overlap_row =
                    std::clamp(
                        current_y,
                        0,
                        overlap_rows - 1
                    );

                // current权重从小到大变化
                current_weight =
                    static_cast<double>(
                        overlap_row + 1
                    ) /
                    static_cast<double>(
                        overlap_rows + 1
                    );

                // previous权重从大到小变化
                previous_weight =
                    1.0 - current_weight;

            } else if (has_previous) {
                // 只有previous存在目标，完全使用previous
                previous_weight = 1.0;

            } else {
                // 只有current存在目标，完全使用current
                current_weight = 1.0;
            }

            /*
             * 6. 处理强度图。
             */
            if (has_previous && has_current) {
                const double first_intensity =
                    previous_intensity.at<double>(
                        previous_y,
                        previous_x
                    );

                const double second_intensity =
                    current_intensity.at<double>(
                        current_y,
                        current_x
                    );

                merged_intensity.at<double>(
                    canvas_y,
                    canvas_x
                ) =
                    previous_weight *
                        first_intensity +
                    current_weight *
                        second_intensity;

            } else if (has_previous) {
                merged_intensity.at<double>(
                    canvas_y,
                    canvas_x
                ) =
                    previous_intensity.at<double>(
                        previous_y,
                        previous_x
                    );

            } else {
                merged_intensity.at<double>(
                    canvas_y,
                    canvas_x
                ) =
                    current_intensity.at<double>(
                        current_y,
                        current_x
                    );
            }

            /*
             * 7. 读取两侧深度并判断有效性。
             */
            const double first_depth =
                has_previous
                    ? previous_depth.at<double>(
                          previous_y,
                          previous_x
                      )
                    : options.invalid_depth;

            const double second_depth =
                has_current
                    ? current_depth.at<double>(
                          current_y,
                          current_x
                      )
                    : options.invalid_depth;

            const bool first_depth_valid =
                has_previous &&
                validDepth(
                    first_depth,
                    options.invalid_depth
                );

            const bool second_depth_valid =
                has_current &&
                validDepth(
                    second_depth,
                    options.invalid_depth
                );

            /*
             * 8. 处理深度图。
             */
            if (first_depth_valid &&
                second_depth_valid) {
                const double depth_difference =
                    std::abs(
                        first_depth -
                        second_depth
                    );

                if (depth_difference <=
                    options.fusion_depth_tolerance) {
                    /*
                     * 两侧深度一致，
                     * 使用与强度图相同的羽化权重。
                     */
                    merged_depth.at<double>(
                        canvas_y,
                        canvas_x
                    ) =
                        previous_weight *
                            first_depth +
                        current_weight *
                            second_depth;

                } else {
                    /*
                     * 两侧深度冲突。
                     *
                     * 不进行平均，选择当前羽化权重更大的一侧，
                     * 防止产生不存在的中间深度。
                     */
                    merged_depth.at<double>(
                        canvas_y,
                        canvas_x
                    ) =
                        previous_weight >=
                                current_weight
                            ? first_depth
                            : second_depth;
                }

            } else if (first_depth_valid) {
                // 只有previous深度有效
                merged_depth.at<double>(
                    canvas_y,
                    canvas_x
                ) = first_depth;

            } else if (second_depth_valid) {
                // 只有current深度有效
                merged_depth.at<double>(
                    canvas_y,
                    canvas_x
                ) = second_depth;

            } else {
                // 两侧深度都无效，保持invalid_depth
                merged_depth.at<double>(
                    canvas_y,
                    canvas_x
                ) = options.invalid_depth;
            }
        }
    }

    /*
     * 9. 生成最终CcdObjectCrop。
     */
    CcdObjectCrop result;

    // 使用previous作为合并结果的基本来源信息
    result.object_info =
        previous.object_info;

    // 合并之后认为物体已经完整
    result.object_info.position =
        CcdObjectPosition::Interior;

    // 合并画布的坐标范围
    result.crop_box =
        cv::Rect(
            global_left,
            0,
            canvas_width,
            canvas_height
        );

    // 转回原图的数据类型
    merged_depth.convertTo(
        result.depth,
        previous.depth.type()
    );

    merged_intensity.convertTo(
        result.intensity,
        previous.intensity.type()
    );

    result.mask =
        merged_mask.clone();

    /*
     * 10. 根据合并后的mask更新物体统计信息。
     */
    result.object_info.area =
        cv::countNonZero(result.mask);

    if (result.object_info.area > 0) {
        // boundingRect得到的是合并图中的局部坐标
        cv::Rect local_box =
            cv::boundingRect(result.mask);

        // 将x坐标恢复成原图横向坐标
        result.object_info.bounding_rect =
            cv::Rect(
                global_left + local_box.x,
                local_box.y,
                local_box.width,
                local_box.height
            );

        // 计算合并mask的质心
        const cv::Moments moments =
            cv::moments(
                result.mask,
                true
            );

        if (moments.m00 > 0.0) {
            result.object_info.center.x =
                global_left +
                moments.m10 / moments.m00;

            result.object_info.center.y =
                moments.m01 / moments.m00;
        }
    }

    return result;
}
