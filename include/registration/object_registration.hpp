#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace object_registration {

struct RegistrationObject {
    std::size_t index = 0;
    std::int64_t timestamp_ms = 0;
    std::filesystem::path image_path;
    std::filesystem::path mask_path;
    std::filesystem::path depth_path;
    cv::Mat mask;
};

struct SequenceMatch {
    std::size_t ccd_position = 0;
    std::size_t xray_position = 0;
    double association_cost = 0.0;
};

struct SequenceAlignment {
    std::vector<SequenceMatch> matches;
    std::vector<std::size_t> unmatched_ccd_positions;
    std::vector<std::size_t> unmatched_xray_positions;
    std::int64_t time_offset_ms = 0;
    double anisotropy_x_over_y = 1.0;
    double area_scale = 1.0;
};

struct RegistrationOptions {
    int timestamp_bin_ms = 100;
    int timestamp_search_radius = 8;
    double gap_cost = 0.95;
    double scale_search_fraction = 0.15;
    double scale_search_step = 0.01;
    double max_rotation_deg = 5.0;
    double rotation_step_deg = 1.0;
    int residual_shift_pixels = 3;
    std::size_t calibration_samples = 80;
    double review_iou = 0.50;
    double low_iou = 0.30;
};

struct VisualizationOptions {
    double overlay_alpha = 0.55;
    std::string colormap = "turbo";
    std::size_t example_count = 8;
    bool include_review_examples = true;
};

struct RegistrationSummary {
    std::string source_run;
    std::filesystem::path output_directory;
    std::size_t ccd_count = 0;
    std::size_t xray_count = 0;
    std::size_t matched_count = 0;
    std::size_t unmatched_ccd_count = 0;
    std::size_t unmatched_xray_count = 0;
    std::size_t review_count = 0;
    std::size_t low_score_count = 0;
    std::size_t example_count = 0;
    std::int64_t time_offset_ms = 0;
    double anisotropy_x_over_y = 1.0;
    double isotropic_scale = 1.0;
    double scale_x = 1.0;
    double scale_y = 1.0;
    double rotation_deg = 0.0;
    double median_iou = 0.0;
};

// Associate two ordered object sequences while allowing objects on either side
// to remain unmatched. This function is also used by the synthetic test.
SequenceAlignment associateObjectSequences(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const RegistrationOptions& options = {}
);

// Register one completed export batch. When run_name is empty, the newest
// completed run shared by results/ccd and results/x is selected.
RegistrationSummary registerExportedRun(
    const std::filesystem::path& results_directory,
    const std::string& run_name = {},
    const RegistrationOptions& options = {},
    const VisualizationOptions& visualization = {}
);

} // namespace object_registration
