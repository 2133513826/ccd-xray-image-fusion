/*
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "common/config.hpp"
#include "io/data_loader.hpp"
#include "ccd/ccd_allmediaprocess.hpp"
#include "ccd/ccd_temporal.hpp"
#include <iostream>
#include <vector>
#include "ccd/ccd_crop.hpp"
#include "x/x_objects.hpp"

// 输出单帧检测得到的所有物体来源信息。
void printCcdObjectSources(
    const std::vector<CcdObjectCrop>& objects
) {
    std::cout << "\nDetected objects: "
              << objects.size() << '\n';

    for (std::size_t i = 0; i < objects.size(); ++i) {
        // 引用当前物体的信息，不复制图像。
        const CcdObjectInfo& info =
            objects[i].object_info;

        std::cout
            << "Object index: " << i
            << ", frame: " << info.file_index
            << ", label: " << info.label
            << ", area: " << info.area
            << '\n';
    }
}












int main(int argc, char* argv[]) {
 if (argc != 2) {
            std::cerr << "Usage: image_fusion <config_file>\n";
            return 1;
        }

//ccd模块读取处理


        // 1. 读取配置和按时间排序的CCD图像。
        const auto config = loadConfig(argv[1]);
        const auto files = inspectCcdFolder(config.ccd_dir.string());
        const auto pairs = loadCcdPairs(files);
        if (pairs.empty()) throw std::runtime_error("No CCD images loaded");
        if (pairs.size() != files.size())
            throw std::runtime_error("Some CCD images failed to load");


       // 1. 建立背景。
const auto background =
    buildCcdBackground(pairs, config.ccd_options);

// 2. 完成单帧处理。objects只定义一次。
const auto objects =
    ccd_allmediaprocess(
        pairs,
        background,
        config.ccd_options
    );

// 3. 打印每个物体来自哪一帧。
printCcdObjectSources(objects);

// 4. 创建连续帧处理参数。
CcdTemporalOptions options;
options.invalid_depth =
    config.ccd_options.invalid_depth;

// 5. 处理跨帧物体。result也只定义一次。
const auto result =
    processCcdTemporalObjects(objects, options);
        options.invalid_depth = config.ccd_options.invalid_depth;
        

    //x模块读取

    auto x_data = loadXrayData(config.x_high_dir, config.x_options);
auto x_objects = processXrayObjects(x_data, config.x_options);



        // 4. 输出ccd数量和每次拼接的判定结果。
        std::cout << "Frames: " << pairs.size()
                  << "\nBefore: " << objects.size()
                  << "\nAfter: " << result.final_ccd_objects.size() << '\n';
        for (const auto& merge : result.merged_info) {
            std::cout << "Pair: " << merge.match.previous_index << " -> "
                      << merge.match.current_index << ", overlap rows: "
                      << merge.decision.best_overlap_rows << ", score: "
                      << merge.decision.best_score << ", mode: "
                      << (merge.decision.mode == CcdFusionMode::RedundancyFusion
                          ? "Feather fusion" : "Direct concat") << '\n';
        }
        // 创建可手动调整大小的窗口，并保持图像宽高比例。
cv::namedWindow(
    "Object intensity",
    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO
);
cv::namedWindow(
    "Object mask",
    cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO
);

// 设置窗口初始大小，单位为屏幕像素。
cv::resizeWindow("Object intensity", 1000, 250);
cv::resizeWindow("Object mask", 1000, 250);

// 将两个窗口上下摆放。
cv::moveWindow("Object intensity", 20, 30);
cv::moveWindow("Object mask", 20, 330);
        // 5. 逐个查看最终对象，按任意键查看下一个。
        for (const auto& object : result.final_ccd_objects) {
            cv::imshow("Object intensity", object.intensity);
            cv::imshow("Object mask", object.mask);
            cv::waitKey(0);
        }
        cv::destroyAllWindows();


        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
*/

/*
#include "common/config.hpp"
#include "io/data_loader.hpp"
#include "x/x_objects.hpp"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// 显示HIGH和mask：任意键继续，Esc停止测试。
bool showXrayPreview(
    const cv::Mat& high,
    const cv::Mat& mask
) {
    cv::Mat gray, image, mask_view;

    // HIGH是16位图，统一映射到8位用于显示。
    // 只改变显示图，不改变原始HIGH数据。
    high.convertTo(gray, CV_8U, 255.0 / 65535.0);
    cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(mask, mask_view, cv::COLOR_GRAY2BGR);

    // 将mask轮廓叠加到HIGH上，观察目标边界。
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        mask.clone(), contours,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE
    );
    cv::drawContours(
        image, contours, -1, cv::Scalar(0, 255, 0), 1
    );

    // 固定画布大小，避免原始图或裁切图撑满屏幕。
    cv::Mat panel(
        400, 900, CV_8UC3, cv::Scalar(35, 35, 35)
    );

    // 将图像等比例缩放，居中放入指定区域。
    auto place = [&](const cv::Mat& source, int left) {
        const double scale = std::min(
            440.0 / source.cols,
            360.0 / source.rows
        );

        const int width =
            std::clamp(cvRound(source.cols * scale), 1, 440);
        const int height =
            std::clamp(cvRound(source.rows * scale), 1, 360);

        cv::Mat resized;
        cv::resize(
            source, resized, cv::Size(width, height),
            0, 0,
            scale < 1 ? cv::INTER_AREA : cv::INTER_NEAREST
        );

        resized.copyTo(panel(cv::Rect(
            left + (440 - width) / 2,
            35 + (360 - height) / 2,
            width, height
        )));
    };

    place(image, 5);
    place(mask_view, 455);

    cv::putText(
        panel, "HIGH + contour", cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255, 255, 255), 1
    );
    cv::putText(
        panel, "Mask", cv::Point(460, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255, 255, 255), 1
    );

    cv::imshow("X-ray test", panel);
    return (cv::waitKey(0) & 0xff) != 27;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: image_fusion <config_file>\n";
        return 1;
    }

    try {
        // 1. 读取配置，再读取并分组X光数据。
        const auto config = loadConfig(argv[1]);
        const auto data = loadXrayData(
            config.x_high_dir, config.x_options
        );

        std::cout << "Groups: " << data.groups.size() << '\n';

        cv::namedWindow(
            "X-ray test",
            cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO
        );
        cv::resizeWindow("X-ray test", 900, 400);

        // 2. 先逐帧检查输入和mask。
        for (std::size_t g = 0; g < data.groups.size(); ++g) {
            std::cout << "\nGroup " << g + 1 << '\n';

            for (const auto& frame : data.groups[g]) {
                std::cout
                    << "Timestamp: " << frame.timestamp_ms
                    << ", labels: " << frame.labels
                    << ", file: "
                    << frame.path.filename().string()
                    << std::endl;

                if (!showXrayPreview(frame.high, frame.mask)) {
                    cv::destroyAllWindows();
                    return 0;
                }
            }
        }

        // 3. 完成组内编号、最大面积选择和裁切。
        const auto objects = processXrayObjects(
            data, config.x_options
        );

        std::cout << "\nFinal objects: "
                  << objects.size() << std::endl;

        // 4. 查看最终结果。
        for (const auto& object : objects) {
            const auto& box = object.bounding_rect;

            std::cout
                << "Object: " << object.object_id
                << ", timestamp: " << object.timestamp_ms
                << ", area: " << cv::countNonZero(object.mask)
                << ", box: ("
                << box.x << ", " << box.y << ", "
                << box.width << ", " << box.height << ")"
                << std::endl;

            if (!showXrayPreview(object.high, object.mask)) {
                break;
            }
        }

        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception& e) {
        cv::destroyAllWindows();
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}




int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: image_fusion <config_file>\n";
        return 1;
    }

    try {
        const auto config = loadConfig(argv[1]);
        const auto data = loadXrayData(
            config.x_high_dir, config.x_options
        );
        const auto objects = processXrayObjects(
            data, config.x_options
        );

        // 先打印全部最终物体的信息。
        std::cout << "Final objects: " << objects.size() << '\n';

        for (const auto& object : objects) {
            std::cout
                << "\nObject ID: " << object.object_id
                << "\nTimestamp (ms): " << object.timestamp_ms
                << "\nBounding rect: " << object.bounding_rect
                << "\nCrop box: " << object.crop_box
                << "\nMask area (pixels): "
                << cv::countNonZero(object.mask)
                << "\nHIGH size: " << object.high.size()
                << "\nMask size: " << object.mask.size()
                << '\n';
        }
        std::cout << std::flush;

        if (objects.empty()) return 0;

        cv::namedWindow(
            "X-ray test",
            cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO
        );
        cv::resizeWindow("X-ray test", 900, 400);

        // 逐个查看最终物体的裁切图和mask。
        for (const auto& object : objects) {
            cv::setWindowTitle(
                "X-ray test",
                "Object " + std::to_string(object.object_id) +
                " | time " + std::to_string(object.timestamp_ms)
            );

            if (!showXrayPreview(object.high, object.mask)) {
                break;
            }
        }

        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception& e) {
        cv::destroyAllWindows();
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}*/
// 项目模块。
#include "common/config.hpp"           // AppConfig、loadConfig。
#include "io/data_loader.hpp"          // CCD和X光数据读取。
#include "ccd/ccd_allmediaprocess.hpp"  // CCD单帧处理。
#include "ccd/ccd_temporal.hpp"         // CCD连续帧处理。
#include "x/x_objects.hpp"              // X光物体提取。

// OpenCV：图像转换、缩放、窗口显示和键盘输入。
#include <opencv2/opencv.hpp>

// C++标准库。
#include <algorithm>  // std::min、std::clamp。
#include <iostream>   // std::cout、std::cerr。
#include <stdexcept>  // std::runtime_error、std::exception。
#include <string>     // std::to_string。
#include <cstddef>    // std::size_t。

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: image_fusion <config_file>\n";
        return 1;
    }
    try {
        const auto config = loadConfig(argv[1]);

        // CCD：读取、建立背景、单帧处理、连续帧处理。
        const auto files = inspectCcdFolder(config.ccd_dir.string());
        const auto pairs = loadCcdPairs(files);
        if (pairs.empty() || pairs.size() != files.size())
            throw std::runtime_error("CCD images are missing");
        const auto background = buildCcdBackground(pairs, config.ccd_options);
        const auto single = ccd_allmediaprocess(pairs, background, config.ccd_options);
        CcdTemporalOptions temporal;
        temporal.invalid_depth = config.ccd_options.invalid_depth;
        const auto result = processCcdTemporalObjects(single, temporal);
        const auto& ccd = result.final_ccd_objects;

        // X：读取、分组、选择每个物体的最大面积观测。
        const auto data = loadXrayData(config.x_high_dir, config.x_options);
        const auto xray = processXrayObjects(data, config.x_options);
        if (ccd.empty() || xray.empty())
            throw std::runtime_error("No final CCD or X-ray objects");

        // 将物体图像和mask并排放入固定大小的画布。
auto preview = [](const cv::Mat& source, const cv::Mat& mask) {
    if (source.empty() || mask.empty() ||
        mask.type() != CV_8UC1 || source.size() != mask.size()) {
        throw std::runtime_error("Invalid object image or mask");
    }

    cv::Mat gray, image_view, mask_view;
    cv::Mat canvas(360, 600, CV_8UC1, cv::Scalar(25));

    // 16位X光转换为8位显示，CCD强度图保持原来的灰度。
    source.convertTo(
        gray, CV_8U,
        source.depth() == CV_16U ? 1.0 / 257.0 : 1.0
    );

    // 两张图使用相同尺寸，便于比较边界。
    const double scale = std::min(
        280.0 / source.cols, 320.0 / source.rows
    );
    const int w = std::clamp(cvRound(source.cols * scale), 1, 280);
    const int h = std::clamp(cvRound(source.rows * scale), 1, 320);

    cv::resize(
        gray, image_view, cv::Size(w, h), 0, 0,
        scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR
    );

    // mask使用最近邻插值，避免出现灰色的过渡像素。
    cv::resize(
        mask, mask_view, cv::Size(w, h),
        0, 0, cv::INTER_NEAREST
    );

    const int top = 35 + (320 - h) / 2;
    image_view.copyTo(canvas(cv::Rect((300 - w) / 2, top, w, h)));
    mask_view.copyTo(canvas(cv::Rect(300 + (300 - w) / 2, top, w, h)));

    cv::putText(canvas, "Image", cv::Point(10, 23),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);
    cv::putText(canvas, "Mask", cv::Point(310, 23),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);

    return canvas;
};

        for (const auto* name : {"CCD", "X-ray"}) {
            cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            cv::resizeWindow(name, 600, 360);
        }
        cv::moveWindow("CCD", 20, 40);
        cv::moveWindow("X-ray", 660, 40);
        std::cout << "CCD: " << ccd.size() << ", X-ray: " << xray.size() << '\n';
        std::size_t ci = 0, xi = 0;

        while (true) {
            // 先同时显示两个窗口，再等待按键。
            cv::imshow("CCD", preview(ccd[ci].intensity, ccd[ci].mask));
cv::imshow("X-ray", preview(xray[xi].high, xray[xi].mask));
            cv::setWindowTitle("CCD", "CCD " + std::to_string(ci+1) + "/" +
                std::to_string(ccd.size()) + " | " +
                std::to_string(ccd[ci].object_info.timestamp_ms));
            cv::setWindowTitle("X-ray", "X " + std::to_string(xray[xi].object_id) + "/" +
                std::to_string(xray.size()) + " | " +
                std::to_string(xray[xi].timestamp_ms));

            const int key = cv::waitKey(0) & 0xff;
            if (key == 27) break;
            if ((key == 'a' || key == 'A') && ci > 0) --ci;
            if ((key == 'd' || key == 'D' || key == ' ') && ci+1 < ccd.size()) ++ci;
            if ((key == 'j' || key == 'J') && xi > 0) --xi;
            if ((key == 'l' || key == 'L' || key == ' ') && xi+1 < xray.size()) ++xi;
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& e) {
        cv::destroyAllWindows();
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}