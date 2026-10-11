#include "x/x_objects.hpp"
#include "common/progress.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

// 一帧中的一次物体观测。
// label是本帧连通域标签；id是关联后得到的组内物体编号。
struct Detection {
    int label = 0;
    int area = 0;
    cv::Rect box;
    cv::Point2d center;
    std::size_t id = 0;
};

// 保存某个物体截至当前面积最大的观测。
struct BestObservation {
    int area = 0;
    XrayObject object;
};

// 从已有mask提取每个物体的信息，并按从上到下排序。
std::vector<Detection> analyzeFrame(
    const XrayFrame& frame,
    cv::Mat& label_map
) {
    if (frame.high.empty() || frame.high.type() != CV_16UC1 ||
        frame.mask.empty() || frame.mask.type() != CV_8UC1 ||
        frame.high.size() != frame.mask.size() || frame.labels < 0) {
        throw std::invalid_argument("Invalid X-ray frame");
    }

    cv::Mat stats, centers;
    const int count = cv::connectedComponentsWithStats(
        frame.mask, label_map, stats, centers, 8, CV_32S
    );

    // 连通域总数包含背景，减去1才是物体数量。
    if (count - 1 != frame.labels)
        throw std::runtime_error("XrayFrame.labels does not match mask");

    std::vector<Detection> objects;

    for (int label = 1; label < count; ++label) {
        Detection d;
        d.label = label;
        d.area = stats.at<int>(label, cv::CC_STAT_AREA);
        d.box = {
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT)
        };
        d.center = {
            centers.at<double>(label, 0),
            centers.at<double>(label, 1)
        };
        objects.push_back(d);
    }

    // y越小越靠上；同高时按照从左到右排序。
    std::sort(objects.begin(), objects.end(),
        [](const auto& a, const auto& b) {
            return std::tie(a.center.y, a.center.x)
                 < std::tie(b.center.y, b.center.x);
        });

    return objects;
}

// 横向IoU：横坐标区间交集长度 / 并集长度。
double horizontalIou(const cv::Rect& a, const cv::Rect& b) {
    const int overlap = std::max(
        0,
        std::min(a.x + a.width, b.x + b.width) -
        std::max(a.x, b.x)
    );

    return static_cast<double>(overlap) /
           (a.width + b.width - overlap);
}

// 为当前帧最上方物体寻找上一帧编号。
// 返回0表示没有找到匹配。
std::size_t findFirstId(
    const Detection& current,
    const std::vector<Detection>& previous,
    double threshold
) {
    std::size_t id = 0;
    double best_score = -1.0;
    bool tied = false;

    for (const auto& old : previous) {
        // 图像y轴向下增大，因此向上移动表现为y减小。
        if (current.center.y >= old.center.y)
            continue;

        const double score = horizontalIou(current.box, old.box);
        if (score < threshold)
            continue;

        if (score > best_score + 1e-12) {
            best_score = score;
            id = old.id;
            tied = false;
        } else if (std::abs(score - best_score) <= 1e-12) {
            tied = true;
        }
    }

    if (tied)
        throw std::runtime_error("Ambiguous top-object match");

    return id;
}

// 从指定观测中裁切原图和当前物体的mask。
XrayObject cropObservation(
    const XrayFrame& frame,
    const cv::Mat& label_map,
    const Detection& d,
    int padding
) {
    XrayObject object;
    object.timestamp_ms = frame.timestamp_ms;
    object.bounding_rect = d.box;

    // 在四周增加边距，不能超出原图。
    const int left = d.box.x - std::min(padding, d.box.x);
    const int top = d.box.y - std::min(padding, d.box.y);
    const int right = d.box.x + d.box.width +
        std::min(padding, frame.high.cols - d.box.x - d.box.width);
    const int bottom = d.box.y + d.box.height +
        std::min(padding, frame.high.rows - d.box.y - d.box.height);

    object.crop_box = {left, top, right - left, bottom - top};
    object.high = frame.high(object.crop_box).clone();

    // 即使裁切框中存在其他物体，也只保留当前标签。
    object.mask = (label_map(object.crop_box) == d.label);

    return object;
}

} // namespace

std::vector<XrayObject> processXrayObjects(
    const XrayData& data,
    const XrayOptions& options
) {
    // 从配置结构读取，后续检查和关联代码可以保持不变。
const double min_horizontal_iou = options.min_horizontal_iou;
    if (options.padding_pixels < 0 ||
        !std::isfinite(min_horizontal_iou) ||
        min_horizontal_iou <= 0 || min_horizontal_iou > 1) {
        throw std::invalid_argument("Invalid X-ray object options");
    }

    std::vector<XrayObject> result;

    for (std::size_t g = 0; g < data.groups.size(); ++g) {
        showProgress("X objects", g, data.groups.size());
        const auto& group = data.groups[g];
        if (group.empty())
            throw std::invalid_argument("Empty X-ray group");

        try {
            // 每帧均只有一个目标时，整组统一使用组内编号1。
            const bool single = std::all_of(
                group.begin(), group.end(),
                [](const auto& frame) {
                    return frame.labels == 1;
                }
            );

            // best[id]保存该编号的最大面积结果；下标0不用。
            std::vector<BestObservation> best(1);
            std::vector<Detection> previous;
            std::size_t largest_id = 0;

            for (std::size_t f = 0; f < group.size(); ++f) {
                const auto& frame = group[f];

                if (f > 0 &&
                    (frame.timestamp_ms <= group[f - 1].timestamp_ms ||
                     frame.high.size() != group.front().high.size())) {
                    throw std::invalid_argument("Invalid frame order or size");
                }

                cv::Mat label_map;
                auto current = analyzeFrame(frame, label_map);
                // 当前帧没有区域通过筛选，不访问front()或分配编号。
if (current.empty()) {
     previous.clear();
    continue;
}

                // 第一帧或单目标组从1开始编号。
                std::size_t first_id = 1;

                if (!single && !previous.empty()) {
                    first_id = findFirstId(
                        current.front(), previous, min_horizontal_iou
                    );

                    // 没有旧匹配，按新进入的一批物体处理。
                    if (first_id == 0)
                        first_id = largest_id + 1;
                }

                // 从最上方编号开始，向下依次编号。
                const auto last_id = first_id + current.size() - 1;

               

                if (last_id >= best.size())
                    best.resize(last_id + 1);

                for (std::size_t k = 0; k < current.size(); ++k) {
                    auto& d = current[k];
                    d.id = first_id + k;

                    auto& record = best[d.id];

                    // 出现更大面积时，更新图像、mask和拍摄时间。
                    // 面积相同则保留先遇到的观测。
                    if (d.area > record.area) {
                        record.area = d.area;
                        record.object = cropObservation(
                            frame, label_map, d, options.padding_pixels
                        );
                    }
                }

               largest_id = std::max(largest_id, last_id);
                previous = std::move(current);
            }

           const auto object_count = largest_id;

            for (std::size_t id = 1; id <= object_count; ++id) {
                if (best[id].area <= 0)
                    throw std::runtime_error("Missing object ID");

                auto object = std::move(best[id].object);

                // 转为整个输出vector中的连续编号。
                object.object_id = result.size() + 1;
                result.push_back(std::move(object));
            }
        } catch (const std::exception& e) {
            // 错误信息中标明组号，方便定位对应图片。
            throw std::runtime_error(
                "X-ray group " + std::to_string(g + 1) + ": " + e.what()
            );
        }
    }

    showProgress("X objects", data.groups.size(), data.groups.size());
    return result;
}
