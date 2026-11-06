#include "registration/registration_config.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace object_registration {

RegistrationConfig loadRegistrationConfig(
    const std::filesystem::path& config_file
) {
    std::ifstream input(config_file);
    if (!input) {
        throw std::runtime_error(
            "Cannot open registration config: " + config_file.string());
    }
    const std::string text{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    cv::FileStorage yaml(text, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!yaml.isOpened()) {
        throw std::runtime_error(
            "Cannot parse registration config: " + config_file.string());
    }

    RegistrationConfig config;
    const auto paths = yaml["paths"];
    if (!paths.isMap() || !paths["results_dir"].isString())
        throw std::runtime_error("Missing paths.results_dir in registration config");
    const auto raw_results = std::filesystem::u8path(
        paths["results_dir"].string());
    config.results_directory = (raw_results.is_absolute()
        ? raw_results
        : std::filesystem::absolute(config_file).parent_path() / raw_results
    ).lexically_normal();

    const auto registration = yaml["registration"];
    const auto visualization = yaml["visualization"];
    if (!registration.isMap())
        throw std::runtime_error("Missing registration mapping");
    if (!visualization.isMap())
        throw std::runtime_error("Missing visualization mapping");

    auto readInt = [](const cv::FileNode& parent, const char* key, int& value) {
        const auto node = parent[key];
        if (!node.isInt())
            throw std::runtime_error(std::string("Missing or invalid integer: ") + key);
        node >> value;
    };
    auto readDouble = [](const cv::FileNode& parent, const char* key, double& value) {
        const auto node = parent[key];
        if (!node.isInt() && !node.isReal())
            throw std::runtime_error(std::string("Missing or invalid number: ") + key);
        node >> value;
    };

    auto& r = config.registration;
    readInt(registration, "timestamp_bin_ms", r.timestamp_bin_ms);
    readInt(registration, "timestamp_search_radius", r.timestamp_search_radius);
    readDouble(registration, "gap_cost", r.gap_cost);
    readDouble(registration, "scale_search_fraction", r.scale_search_fraction);
    readDouble(registration, "scale_search_step", r.scale_search_step);
    readDouble(registration, "max_rotation_deg", r.max_rotation_deg);
    readDouble(registration, "rotation_step_deg", r.rotation_step_deg);
    readInt(registration, "residual_shift_pixels", r.residual_shift_pixels);
    int calibration_samples = 0;
    readInt(registration, "calibration_samples", calibration_samples);
    if (calibration_samples <= 0)
        throw std::runtime_error("calibration_samples must be positive");
    r.calibration_samples = static_cast<std::size_t>(calibration_samples);
    readDouble(registration, "review_iou", r.review_iou);
    readDouble(registration, "low_iou", r.low_iou);

    auto& v = config.visualization;
    readDouble(visualization, "overlay_alpha", v.overlay_alpha);
    if (!visualization["colormap"].isString())
        throw std::runtime_error("Missing or invalid visualization.colormap");
    v.colormap = visualization["colormap"].string();
    int example_count = 0;
    readInt(visualization, "example_count", example_count);
    if (example_count < 0)
        throw std::runtime_error("example_count cannot be negative");
    v.example_count = static_cast<std::size_t>(example_count);
    int include_review = 0;
    readInt(visualization, "include_review_examples", include_review);
    if (include_review != 0 && include_review != 1)
        throw std::runtime_error("include_review_examples must be 0 or 1");
    v.include_review_examples = include_review != 0;

    if (!(r.gap_cost > 0.0) || r.timestamp_bin_ms <= 0 ||
        r.timestamp_search_radius < 0 || !(r.scale_search_fraction >= 0.0) ||
        !(r.scale_search_step > 0.0) || !(r.max_rotation_deg >= 0.0) ||
        !(r.rotation_step_deg > 0.0) || r.residual_shift_pixels < 0 ||
        !(r.low_iou >= 0.0 && r.low_iou <= r.review_iou && r.review_iou <= 1.0)) {
        throw std::runtime_error("Registration options are outside their valid ranges");
    }
    if (!std::isfinite(v.overlay_alpha) || v.overlay_alpha < 0.0 ||
        v.overlay_alpha > 1.0 || v.colormap.empty()) {
        throw std::runtime_error("Visualization options are outside their valid ranges");
    }
    return config;
}

} // namespace object_registration
