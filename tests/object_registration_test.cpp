#include "registration/object_registration.hpp"
#include "registration/registration_config.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using object_registration::RegistrationObject;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

cv::Mat makeXMask(int kind) {
    cv::Mat mask(100, 100, CV_8UC1, cv::Scalar(0));
    if (kind == 0) cv::ellipse(mask, {50, 50}, {27, 20}, 0, 0, 360, 255, -1);
    if (kind == 1) cv::rectangle(mask, {22, 30}, {78, 68}, 255, -1);
    if (kind == 2) {
        const std::vector<cv::Point> points{{18, 72}, {38, 20}, {82, 32}, {72, 77}};
        cv::fillConvexPoly(mask, points, 255);
    }
    if (kind == 3) {
        cv::circle(mask, {37, 50}, 23, 255, -1);
        cv::circle(mask, {67, 50}, 17, 255, -1);
    }
    if (kind == 4) {
        const std::vector<cv::Point> points{{15, 40}, {60, 18}, {85, 58}, {42, 82}};
        cv::fillConvexPoly(mask, points, 255);
    }
    return mask;
}

cv::Mat makeCcdMask(const cv::Mat& xmask) {
    cv::Mat result;
    cv::resize(xmask, result, cv::Size(), 2.0, 0.5, cv::INTER_NEAREST);
    return result;
}

RegistrationObject object(std::size_t index, std::int64_t timestamp, cv::Mat mask) {
    RegistrationObject result;
    result.index = index;
    result.timestamp_ms = timestamp;
    result.mask = std::move(mask);
    return result;
}

} // namespace

int main() {
    try {
        std::vector<RegistrationObject> ccd;
        std::vector<RegistrationObject> xray;
        const std::vector<int> kinds{0, 1, 2, 3};
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            const auto xmask = makeXMask(kinds[i]);
            ccd.push_back(object(i + 1, 1000 + static_cast<std::int64_t>(i) * 1300,
                                 makeCcdMask(xmask)));
        }
        xray.push_back(object(1, 11000, makeXMask(0)));
        xray.push_back(object(2, 12300, makeXMask(1)));
        xray.push_back(object(3, 12900, makeXMask(4))); // X-only object.
        xray.push_back(object(4, 13600, makeXMask(2)));
        xray.push_back(object(5, 14900, makeXMask(3)));

        object_registration::RegistrationOptions options;
        options.timestamp_search_radius = 3;
        const auto result = object_registration::associateObjectSequences(
            ccd, xray, options);
        require(result.matches.size() == 4, "Expected four matches");
        require(result.unmatched_ccd_positions.empty(), "Unexpected unmatched CCD object");
        require(result.unmatched_xray_positions.size() == 1,
                "Expected one unmatched X-ray object");
        require(result.unmatched_xray_positions.front() == 2,
                "Wrong X-ray object was left unmatched");
        for (std::size_t i = 0; i < result.matches.size(); ++i) {
            require(result.matches[i].ccd_position == i, "CCD sequence order changed");
            const std::size_t expected_x = i < 2 ? i : i + 1;
            require(result.matches[i].xray_position == expected_x,
                    "Incorrect sequence association");
        }
        require(std::llabs(result.time_offset_ms - 10000) <= 1,
                "Incorrect timestamp offset");
        require(std::abs(result.anisotropy_x_over_y - 0.25) < 0.04,
                "Incorrect anisotropy estimate");

        const auto config_root = std::filesystem::current_path() /
            "registration_config_test_data";
        std::filesystem::create_directories(config_root / "results");
        const auto config_path = config_root / "registration.yaml";
        std::ofstream config_file(config_path);
        config_file
            << "%YAML:1.0\n---\n"
            << "paths:\n  results_dir: \"results\"\n"
            << "registration:\n"
            << "  timestamp_bin_ms: 80\n"
            << "  timestamp_search_radius: 6\n"
            << "  gap_cost: 1.1\n"
            << "  scale_search_fraction: 0.12\n"
            << "  scale_search_step: 0.02\n"
            << "  max_rotation_deg: 4.0\n"
            << "  rotation_step_deg: 0.5\n"
            << "  residual_shift_pixels: 2\n"
            << "  calibration_samples: 40\n"
            << "  review_iou: 0.55\n"
            << "  low_iou: 0.25\n"
            << "visualization:\n"
            << "  overlay_alpha: 0.6\n"
            << "  colormap: \"viridis\"\n"
            << "  example_count: 6\n"
            << "  include_review_examples: 1\n";
        config_file.close();
        require(static_cast<bool>(config_file), "Cannot create config test input");
        const auto config = object_registration::loadRegistrationConfig(config_path);
        require(config.registration.timestamp_bin_ms == 80,
                "Config integer was not loaded");
        require(std::abs(config.registration.gap_cost - 1.1) < 1e-9,
                "Config number was not loaded");
        require(config.visualization.colormap == "viridis" &&
                config.visualization.example_count == 6,
                "Visualization config was not loaded");
        require(config.results_directory == (config_root / "results").lexically_normal(),
                "Relative results path was not resolved from the config directory");
        std::filesystem::remove_all(config_root);
        std::cout << "object_registration_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "object_registration_test failed: " << error.what() << '\n';
        return 1;
    }
}
