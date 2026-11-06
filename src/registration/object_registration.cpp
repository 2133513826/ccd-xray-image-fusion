#include "registration/object_registration.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace object_registration {
namespace {

struct ObjectFeature {
    double area = 0.0;
    double aspect = 1.0;
    double solidity = 1.0;
    cv::Point2d center;
};

struct PairArtifact {
    std::size_t pair_index = 0;
    std::size_t ccd_index = 0;
    std::size_t xray_index = 0;
    double iou = 0.0;
    std::string status;
    std::string stem;
    std::filesystem::path original_ccd_image;
    std::filesystem::path original_xray_image;
    std::filesystem::path registered_ccd_image;
    std::filesystem::path transparent_fusion;
    std::filesystem::path false_color_fusion;
};

cv::Mat readImage(const std::filesystem::path& path, int flags) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open image: " + path.string());
    const std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1,
                          const_cast<unsigned char*>(bytes.data()));
    cv::Mat image = cv::imdecode(encoded, flags);
    if (image.empty()) throw std::runtime_error("Cannot decode image: " + path.string());
    return image;
}

void writeImage(const std::filesystem::path& path, const cv::Mat& image) {
    std::vector<unsigned char> bytes;
    if (image.empty() || !cv::imencode(path.extension().string(), image, bytes))
        throw std::runtime_error("Cannot encode image: " + path.string());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Cannot write image: " + path.string());
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("Cannot calculate an empty median");
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0) return upper;
    std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return (upper + values[middle - 1]) * 0.5;
}

std::int64_t medianInt64(const std::vector<std::int64_t>& input) {
    std::vector<double> values;
    values.reserve(input.size());
    for (const auto value : input) values.push_back(static_cast<double>(value));
    return static_cast<std::int64_t>(std::llround(median(std::move(values))));
}

ObjectFeature featureOf(const cv::Mat& input) {
    if (input.empty() || input.type() != CV_8UC1)
        throw std::invalid_argument("Registration masks must be CV_8UC1");
    cv::Mat mask;
    cv::compare(input, 0, mask, cv::CMP_GT);
    const int area = cv::countNonZero(mask);
    if (area == 0) throw std::invalid_argument("Registration mask is empty");

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    const cv::Rect box = cv::boundingRect(points);
    const cv::Moments moments = cv::moments(mask, true);

    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);
    const double hull_area = std::max(1.0, cv::contourArea(hull));

    ObjectFeature result;
    result.area = static_cast<double>(area);
    result.aspect = static_cast<double>(box.width) / box.height;
    result.solidity = std::min(1.0, result.area / hull_area);
    result.center = cv::Point2d(moments.m10 / moments.m00,
                                moments.m01 / moments.m00);
    return result;
}

std::vector<ObjectFeature> featuresOf(const std::vector<RegistrationObject>& objects) {
    std::vector<ObjectFeature> result;
    result.reserve(objects.size());
    for (const auto& object : objects) result.push_back(featureOf(object.mask));
    return result;
}

double distributionRatio(
    const std::vector<ObjectFeature>& numerator,
    const std::vector<ObjectFeature>& denominator,
    double ObjectFeature::*member
) {
    std::vector<double> first, second;
    first.reserve(numerator.size());
    second.reserve(denominator.size());
    for (const auto& value : numerator) first.push_back(std::log(value.*member));
    for (const auto& value : denominator) second.push_back(std::log(value.*member));
    return std::exp(median(std::move(first)) - median(std::move(second)));
}

std::int64_t estimateTimeOffset(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const RegistrationOptions& options
) {
    if (ccd.empty() || xray.empty()) throw std::invalid_argument("Empty object sequence");
    if (options.timestamp_bin_ms <= 0 || options.timestamp_search_radius < 0)
        throw std::invalid_argument("Invalid timestamp search options");

    std::unordered_map<std::int64_t, int> bins;
    for (std::size_t i = 0; i < ccd.size(); ++i) {
        const double fraction = ccd.size() == 1 ? 0.0
            : static_cast<double>(i) / static_cast<double>(ccd.size() - 1);
        const int expected = static_cast<int>(std::llround(
            fraction * static_cast<double>(xray.size() - 1)));
        const int begin = std::max(0, expected - options.timestamp_search_radius);
        const int end = std::min(static_cast<int>(xray.size()) - 1,
                                 expected + options.timestamp_search_radius);
        for (int j = begin; j <= end; ++j) {
            const std::int64_t delta = xray[j].timestamp_ms - ccd[i].timestamp_ms;
            const auto bin = static_cast<std::int64_t>(std::llround(
                static_cast<double>(delta) / options.timestamp_bin_ms));
            ++bins[bin];
        }
    }

    std::int64_t best_bin = 0;
    int best_count = -1;
    for (const auto& entry : bins) {
        int smoothed = 0;
        for (int offset = -3; offset <= 3; ++offset) {
            const auto found = bins.find(entry.first + offset);
            if (found != bins.end()) smoothed += found->second;
        }
        if (smoothed > best_count) {
            best_count = smoothed;
            best_bin = entry.first;
        }
    }

    const std::int64_t center = best_bin * options.timestamp_bin_ms;
    const std::int64_t radius = 5LL * options.timestamp_bin_ms;
    std::vector<std::int64_t> candidates;
    for (std::size_t i = 0; i < ccd.size(); ++i) {
        const double fraction = ccd.size() == 1 ? 0.0
            : static_cast<double>(i) / static_cast<double>(ccd.size() - 1);
        const int expected = static_cast<int>(std::llround(
            fraction * static_cast<double>(xray.size() - 1)));
        const int begin = std::max(0, expected - options.timestamp_search_radius);
        const int end = std::min(static_cast<int>(xray.size()) - 1,
                                 expected + options.timestamp_search_radius);
        for (int j = begin; j <= end; ++j) {
            const std::int64_t delta = xray[j].timestamp_ms - ccd[i].timestamp_ms;
            if (std::llabs(delta - center) <= radius) candidates.push_back(delta);
        }
    }
    if (candidates.empty()) throw std::runtime_error("Cannot estimate camera time offset");
    return medianInt64(candidates);
}

double associationCost(
    const RegistrationObject& ccd,
    const RegistrationObject& xray,
    const ObjectFeature& cf,
    const ObjectFeature& xf,
    std::int64_t time_offset,
    double anisotropy,
    double area_scale
) {
    const double time_difference = std::abs(static_cast<double>(
        xray.timestamp_ms - ccd.timestamp_ms - time_offset));
    const double time_cost = std::min(4.0, time_difference / 500.0);
    const double aspect_cost = std::min(4.0,
        std::abs(std::log((cf.aspect * anisotropy) / xf.aspect)) / 0.20);
    const double area_cost = std::min(4.0,
        std::abs(std::log((cf.area * area_scale) / xf.area)) / 0.35);
    const double solidity_cost = std::min(4.0,
        std::abs(cf.solidity - xf.solidity) / 0.12);
    return 0.55 * time_cost + 0.20 * aspect_cost +
           0.20 * area_cost + 0.05 * solidity_cost;
}

SequenceAlignment dynamicAlignment(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const std::vector<ObjectFeature>& cf,
    const std::vector<ObjectFeature>& xf,
    std::int64_t time_offset,
    double anisotropy,
    double area_scale,
    double gap_cost
) {
    const std::size_t rows = ccd.size() + 1;
    const std::size_t cols = xray.size() + 1;
    std::vector<double> cost(rows * cols, std::numeric_limits<double>::infinity());
    std::vector<unsigned char> action(rows * cols, 0);
    const auto at = [cols](std::size_t i, std::size_t j) { return i * cols + j; };
    cost[0] = 0.0;
    for (std::size_t i = 1; i < rows; ++i) {
        cost[at(i, 0)] = cost[at(i - 1, 0)] + gap_cost;
        action[at(i, 0)] = 'C';
    }
    for (std::size_t j = 1; j < cols; ++j) {
        cost[at(0, j)] = cost[at(0, j - 1)] + gap_cost;
        action[at(0, j)] = 'X';
    }

    for (std::size_t i = 1; i < rows; ++i) {
        for (std::size_t j = 1; j < cols; ++j) {
            const double pair_cost = associationCost(
                ccd[i - 1], xray[j - 1], cf[i - 1], xf[j - 1],
                time_offset, anisotropy, area_scale);
            const double match = cost[at(i - 1, j - 1)] + pair_cost;
            const double skip_ccd = cost[at(i - 1, j)] + gap_cost;
            const double skip_xray = cost[at(i, j - 1)] + gap_cost;
            if (match <= skip_ccd && match <= skip_xray) {
                cost[at(i, j)] = match;
                action[at(i, j)] = 'M';
            } else if (skip_ccd <= skip_xray) {
                cost[at(i, j)] = skip_ccd;
                action[at(i, j)] = 'C';
            } else {
                cost[at(i, j)] = skip_xray;
                action[at(i, j)] = 'X';
            }
        }
    }

    SequenceAlignment result;
    result.time_offset_ms = time_offset;
    result.anisotropy_x_over_y = anisotropy;
    result.area_scale = area_scale;
    std::size_t i = ccd.size(), j = xray.size();
    while (i != 0 || j != 0) {
        const unsigned char step = action[at(i, j)];
        if (step == 'M') {
            const double pair_cost = associationCost(
                ccd[i - 1], xray[j - 1], cf[i - 1], xf[j - 1],
                time_offset, anisotropy, area_scale);
            result.matches.push_back({i - 1, j - 1, pair_cost});
            --i; --j;
        } else if (step == 'C') {
            result.unmatched_ccd_positions.push_back(i - 1);
            --i;
        } else if (step == 'X') {
            result.unmatched_xray_positions.push_back(j - 1);
            --j;
        } else {
            throw std::runtime_error("Invalid sequence alignment backtrack");
        }
    }
    std::reverse(result.matches.begin(), result.matches.end());
    std::reverse(result.unmatched_ccd_positions.begin(),
                 result.unmatched_ccd_positions.end());
    std::reverse(result.unmatched_xray_positions.begin(),
                 result.unmatched_xray_positions.end());
    return result;
}

cv::Matx23d transformMatrix(
    const ObjectFeature& source,
    const ObjectFeature& target,
    double anisotropy,
    double isotropic_scale,
    double rotation_deg,
    int shift_x = 0,
    int shift_y = 0
) {
    const double radians = rotation_deg * CV_PI / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    const double sx = anisotropy * isotropic_scale;
    const double sy = isotropic_scale;
    const double a00 = cosine * sx;
    const double a01 = -sine * sy;
    const double a10 = sine * sx;
    const double a11 = cosine * sy;
    const double tx = target.center.x + shift_x -
                      a00 * source.center.x - a01 * source.center.y;
    const double ty = target.center.y + shift_y -
                      a10 * source.center.x - a11 * source.center.y;
    return cv::Matx23d(a00, a01, tx, a10, a11, ty);
}

cv::Mat warpMask(const cv::Mat& mask, const cv::Matx23d& matrix, cv::Size size) {
    cv::Mat warped;
    cv::warpAffine(mask, warped, cv::Mat(matrix), size, cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::compare(warped, 0, warped, cv::CMP_GT);
    return warped;
}

std::pair<double, double> overlapScores(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat intersection, combined;
    cv::bitwise_and(first, second, intersection);
    cv::bitwise_or(first, second, combined);
    const double intersection_count = cv::countNonZero(intersection);
    const double union_count = cv::countNonZero(combined);
    const double total = cv::countNonZero(first) + cv::countNonZero(second);
    return {union_count > 0.0 ? intersection_count / union_count : 0.0,
            total > 0.0 ? 2.0 * intersection_count / total : 0.0};
}

std::vector<std::size_t> calibrationIndices(std::size_t count, std::size_t limit) {
    if (count == 0) return {};
    const std::size_t selected = std::min(count, std::max<std::size_t>(1, limit));
    std::vector<std::size_t> result;
    result.reserve(selected);
    if (selected == 1) return {count / 2};
    for (std::size_t i = 0; i < selected; ++i) {
        result.push_back(static_cast<std::size_t>(std::llround(
            static_cast<double>(i) * (count - 1) / (selected - 1))));
    }
    return result;
}

double medianCalibrationIou(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const std::vector<ObjectFeature>& cf,
    const std::vector<ObjectFeature>& xf,
    const SequenceAlignment& alignment,
    const std::vector<std::size_t>& selected,
    double anisotropy,
    double scale,
    double rotation
) {
    std::vector<double> values;
    values.reserve(selected.size());
    for (const std::size_t selected_position : selected) {
        const auto& pair = alignment.matches[selected_position];
        const auto matrix = transformMatrix(
            cf[pair.ccd_position], xf[pair.xray_position],
            anisotropy, scale, rotation);
        const cv::Mat warped = warpMask(
            ccd[pair.ccd_position].mask, matrix,
            xray[pair.xray_position].mask.size());
        values.push_back(overlapScores(warped, xray[pair.xray_position].mask).first);
    }
    return median(std::move(values));
}

std::pair<double, double> calibrateScaleAndRotation(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const std::vector<ObjectFeature>& cf,
    const std::vector<ObjectFeature>& xf,
    const SequenceAlignment& alignment,
    const RegistrationOptions& options
) {
    if (alignment.matches.empty()) throw std::runtime_error("No pairs for calibration");
    const auto selected = calibrationIndices(
        alignment.matches.size(), options.calibration_samples);
    const double initial = std::sqrt(
        alignment.area_scale / alignment.anisotropy_x_over_y);
    const double minimum = initial * (1.0 - options.scale_search_fraction);
    const double maximum = initial * (1.0 + options.scale_search_fraction);
    double best_scale = initial;
    double best_score = -1.0;
    for (double scale = minimum; scale <= maximum + 1e-9;
         scale += options.scale_search_step) {
        const double score = medianCalibrationIou(
            ccd, xray, cf, xf, alignment, selected,
            alignment.anisotropy_x_over_y, scale, 0.0);
        if (score > best_score) {
            best_score = score;
            best_scale = scale;
        }
    }

    double best_rotation = 0.0;
    best_score = -1.0;
    for (double angle = -options.max_rotation_deg;
         angle <= options.max_rotation_deg + 1e-9;
         angle += options.rotation_step_deg) {
        const double score = medianCalibrationIou(
            ccd, xray, cf, xf, alignment, selected,
            alignment.anisotropy_x_over_y, best_scale, angle);
        if (score > best_score) {
            best_score = score;
            best_rotation = angle;
        }
    }
    return {best_scale, best_rotation};
}

std::pair<std::size_t, std::int64_t> parseObjectName(
    const std::filesystem::path& path
) {
    const std::string name = path.stem().string();
    const std::size_t separator = name.find('_');
    if (separator == std::string::npos)
        throw std::runtime_error("Invalid object filename: " + path.string());
    try {
        return {static_cast<std::size_t>(std::stoull(name.substr(0, separator))),
                std::stoll(name.substr(separator + 1))};
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid object filename: " + path.string());
    }
}

std::vector<RegistrationObject> loadObjects(
    const std::filesystem::path& root,
    bool ccd
) {
    const auto mask_directory = root / "masks";
    if (!std::filesystem::is_directory(mask_directory))
        throw std::runtime_error("Missing masks directory: " + mask_directory.string());
    std::vector<std::filesystem::path> mask_paths;
    for (const auto& entry : std::filesystem::directory_iterator(mask_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            mask_paths.push_back(entry.path());
    }
    std::sort(mask_paths.begin(), mask_paths.end());
    if (mask_paths.empty()) throw std::runtime_error("No exported masks in " + root.string());

    std::vector<RegistrationObject> result;
    result.reserve(mask_paths.size());
    for (const auto& mask_path : mask_paths) {
        const auto parsed = parseObjectName(mask_path);
        RegistrationObject object;
        object.index = parsed.first;
        object.timestamp_ms = parsed.second;
        object.mask_path = mask_path;
        object.image_path = root / "images" / mask_path.filename();
        object.depth_path = ccd
            ? root / "depth" / (mask_path.stem().string() + ".tiff")
            : std::filesystem::path();
        if (!std::filesystem::is_regular_file(object.image_path) ||
            (ccd && !std::filesystem::is_regular_file(object.depth_path))) {
            throw std::runtime_error("Incomplete exported object: " + mask_path.string());
        }
        object.mask = readImage(mask_path, cv::IMREAD_GRAYSCALE);
        cv::compare(object.mask, 0, object.mask, cv::CMP_GT);
        if (cv::countNonZero(object.mask) == 0)
            throw std::runtime_error("Empty exported mask: " + mask_path.string());
        result.push_back(std::move(object));
    }
    return result;
}

std::string selectRun(const std::filesystem::path& results, std::string requested) {
    if (!requested.empty() && requested.rfind("run_", 0) != 0)
        requested = "run_" + requested;
    auto complete = [&](const std::string& name) {
        return std::filesystem::is_regular_file(results / "ccd" / name / "COMPLETE.txt") &&
               std::filesystem::is_regular_file(results / "x" / name / "COMPLETE.txt");
    };
    if (!requested.empty()) {
        if (!complete(requested))
            throw std::runtime_error("Run is not complete on both sides: " + requested);
        return requested;
    }

    std::vector<std::string> candidates;
    const auto ccd_root = results / "ccd";
    if (!std::filesystem::is_directory(ccd_root))
        throw std::runtime_error("Missing results/ccd directory");
    for (const auto& entry : std::filesystem::directory_iterator(ccd_root)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind("run_", 0) == 0 && complete(name))
            candidates.push_back(name);
    }
    if (candidates.empty()) throw std::runtime_error("No common completed CCD/X run");
    return *std::max_element(candidates.begin(), candidates.end());
}

std::filesystem::path uniqueOutputPath(
    const std::filesystem::path& registration_root,
    const std::string& run
) {
    std::filesystem::path candidate = registration_root / run;
    auto occupied = [](const std::filesystem::path& path) {
        return std::filesystem::exists(path) ||
               std::filesystem::exists(std::filesystem::path(path.string() + ".partial"));
    };
    for (int suffix = 2; occupied(candidate); ++suffix)
        candidate = registration_root / (run + "_" + std::to_string(suffix));
    return candidate;
}

std::string pairStem(std::size_t pair, std::size_t ccd, std::size_t xray) {
    std::ostringstream text;
    text << std::setfill('0') << std::setw(6) << pair
         << "_c" << std::setw(6) << ccd
         << "_x" << std::setw(6) << xray;
    return text.str();
}

std::string csvText(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        escaped.push_back(character);
        if (character == '"') escaped.push_back('"');
    }
    escaped.push_back('"');
    return escaped;
}

cv::Mat displayGray(const cv::Mat& image) {
    cv::Mat gray;
    if (image.type() == CV_8UC1) {
        gray = image.clone();
    } else {
        cv::normalize(image, gray, 0, 220, cv::NORM_MINMAX, CV_8U);
    }
    return gray;
}

int colormapCode(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (name == "turbo") return cv::COLORMAP_TURBO;
    if (name == "jet") return cv::COLORMAP_JET;
    if (name == "viridis") return cv::COLORMAP_VIRIDIS;
    if (name == "plasma") return cv::COLORMAP_PLASMA;
    if (name == "magma") return cv::COLORMAP_MAGMA;
    if (name == "inferno") return cv::COLORMAP_INFERNO;
    if (name == "hot") return cv::COLORMAP_HOT;
    if (name == "bone") return cv::COLORMAP_BONE;
    throw std::invalid_argument("Unsupported visualization colormap: " + name);
}

cv::Mat normalizeMaskedGray(const cv::Mat& image, const cv::Mat& mask) {
    if (image.empty() || image.channels() != 1 || mask.type() != CV_8UC1 ||
        image.size() != mask.size()) {
        throw std::invalid_argument("Invalid image or mask for visualization");
    }
    double minimum = 0.0, maximum = 0.0;
    cv::minMaxLoc(image, &minimum, &maximum, nullptr, nullptr, mask);
    cv::Mat result(image.size(), CV_8UC1, cv::Scalar(0));
    if (maximum > minimum) {
        cv::Mat normalized;
        image.convertTo(normalized, CV_8U,
                        255.0 / (maximum - minimum),
                        -minimum * 255.0 / (maximum - minimum));
        normalized.copyTo(result, mask);
    } else {
        result.setTo(180, mask);
    }
    return result;
}

cv::Mat transparentFusion(
    const cv::Mat& xray_image,
    const cv::Mat& xray_mask,
    const cv::Mat& ccd_image,
    const cv::Mat& ccd_mask,
    const VisualizationOptions& options
) {
    const cv::Mat xray_gray = normalizeMaskedGray(xray_image, xray_mask);
    const cv::Mat ccd_gray = normalizeMaskedGray(ccd_image, ccd_mask);
    cv::Mat base, pseudocolor;
    cv::cvtColor(xray_gray, base, cv::COLOR_GRAY2BGR);
    cv::applyColorMap(ccd_gray, pseudocolor, colormapCode(options.colormap));
    pseudocolor.setTo(cv::Scalar(0, 0, 0), ccd_mask == 0);
    cv::Mat blended;
    cv::addWeighted(base, 1.0 - options.overlay_alpha,
                    pseudocolor, options.overlay_alpha, 0.0, blended);
    blended.copyTo(base, ccd_mask);
    return base;
}

cv::Mat falseColorFusion(
    const cv::Mat& xray_image,
    const cv::Mat& xray_mask,
    const cv::Mat& ccd_image,
    const cv::Mat& ccd_mask
) {
    const cv::Mat xray_gray = normalizeMaskedGray(xray_image, xray_mask);
    const cv::Mat ccd_gray = normalizeMaskedGray(ccd_image, ccd_mask);
    cv::Mat result(xray_image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            const unsigned char xr = xray_gray.at<unsigned char>(y, x);
            const unsigned char cc = ccd_gray.at<unsigned char>(y, x);
            result.at<cv::Vec3b>(y, x) = cv::Vec3b(xr, cc, xr);
        }
    }
    return result;
}

cv::Mat agreementOverlay(
    const cv::Mat& xray_image,
    const cv::Mat& ccd_mask,
    const cv::Mat& xray_mask
) {
    cv::Mat gray = displayGray(xray_image);
    cv::Mat result;
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            const bool c = ccd_mask.at<unsigned char>(y, x) != 0;
            const bool r = xray_mask.at<unsigned char>(y, x) != 0;
            if (!c && !r) continue;
            const cv::Vec3b color = c && r ? cv::Vec3b(0, 230, 230)
                : (c ? cv::Vec3b(0, 220, 0) : cv::Vec3b(0, 0, 230));
            cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
            for (int channel = 0; channel < 3; ++channel)
                pixel[channel] = cv::saturate_cast<unsigned char>(
                    0.35 * pixel[channel] + 0.65 * color[channel]);
        }
    }
    return result;
}

void placePreview(
    cv::Mat& panel,
    const cv::Mat& input,
    int left,
    const std::string& title
) {
    cv::Mat image;
    if (input.channels() == 1) cv::cvtColor(displayGray(input), image, cv::COLOR_GRAY2BGR);
    else image = input;
    const double scale = std::min(220.0 / image.cols, 210.0 / image.rows);
    const int width = std::max(1, cvRound(image.cols * scale));
    const int height = std::max(1, cvRound(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0, 0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_NEAREST);
    resized.copyTo(panel(cv::Rect(left + (240 - width) / 2,
                                  42 + (210 - height) / 2,
                                  width, height)));
    cv::putText(panel, title, cv::Point(left + 8, 27),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(245, 245, 245), 1,
                cv::LINE_AA);
}

cv::Mat makePreview(
    const cv::Mat& registered_ccd,
    const cv::Mat& xray,
    const cv::Mat& overlay,
    const std::string& status,
    double iou
) {
    cv::Mat panel(280, 740, CV_8UC3, cv::Scalar(28, 28, 28));
    placePreview(panel, registered_ccd, 0, "Registered CCD");
    placePreview(panel, xray, 250, "X-ray");
    placePreview(panel, overlay, 500, "Mask agreement");
    std::ostringstream label;
    label << status << "  IoU=" << std::fixed << std::setprecision(3) << iou;
    cv::putText(panel, label.str(), cv::Point(12, 272),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(230, 230, 230), 1,
                cv::LINE_AA);
    return panel;
}

cv::Mat makeExampleSheet(const PairArtifact& artifact) {
    const cv::Mat original_ccd = readImage(
        artifact.original_ccd_image, cv::IMREAD_UNCHANGED);
    const cv::Mat original_xray = readImage(
        artifact.original_xray_image, cv::IMREAD_UNCHANGED);
    const cv::Mat registered_ccd = readImage(
        artifact.registered_ccd_image, cv::IMREAD_UNCHANGED);
    const cv::Mat transparent = readImage(
        artifact.transparent_fusion, cv::IMREAD_COLOR);
    const cv::Mat false_color = readImage(
        artifact.false_color_fusion, cv::IMREAD_COLOR);

    cv::Mat panel(310, 1240, CV_8UC3, cv::Scalar(28, 28, 28));
    placePreview(panel, original_ccd, 0, "Original CCD");
    placePreview(panel, original_xray, 250, "Original X-ray");
    placePreview(panel, registered_ccd, 500, "Registered CCD");
    placePreview(panel, transparent, 750, "Transparent fusion");
    placePreview(panel, false_color, 1000, "False color fusion");
    std::ostringstream label;
    label << "pair=" << artifact.pair_index
          << "  status=" << artifact.status
          << "  IoU=" << std::fixed << std::setprecision(3) << artifact.iou;
    cv::putText(panel, label.str(), cv::Point(12, 302),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    return panel;
}

std::vector<std::size_t> selectExamples(
    const std::vector<PairArtifact>& artifacts,
    const VisualizationOptions& options
) {
    const std::size_t target = std::min(options.example_count, artifacts.size());
    std::set<std::size_t> selected;
    if (options.include_review_examples) {
        for (std::size_t i = 0; i < artifacts.size() && selected.size() < target; ++i) {
            if (artifacts[i].status != "good") selected.insert(i);
        }
    }
    const std::size_t remaining = target - selected.size();
    auto insertNearest = [&](std::size_t position) {
        for (std::size_t distance = 0; distance < artifacts.size(); ++distance) {
            if (position + distance < artifacts.size() &&
                selected.insert(position + distance).second) return;
            if (distance <= position &&
                selected.insert(position - distance).second) return;
        }
    };
    for (std::size_t i = 0; i < remaining; ++i) {
        const std::size_t position = remaining == 1 ? artifacts.size() / 2
            : static_cast<std::size_t>(std::llround(
                static_cast<double>(i) * (artifacts.size() - 1) / (remaining - 1)));
        insertNearest(position);
    }
    for (std::size_t i = 0; i < artifacts.size() && selected.size() < target; ++i)
        selected.insert(i);
    return {selected.begin(), selected.end()};
}

std::size_t writeExamples(
    const std::filesystem::path& temporary_root,
    const std::vector<PairArtifact>& artifacts,
    const VisualizationOptions& options
) {
    const auto selected = selectExamples(artifacts, options);
    if (selected.empty()) return 0;
    const auto examples_directory = temporary_root / "examples";
    std::filesystem::create_directories(examples_directory);
    std::ofstream metadata(examples_directory / "examples.csv");
    if (!metadata) throw std::runtime_error("Cannot create examples metadata");
    metadata << "example_index,pair_index,ccd_index,x_index,status,iou,reason,image_path\n";
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const auto& artifact = artifacts[selected[i]];
        std::ostringstream filename;
        filename << std::setfill('0') << std::setw(2) << i + 1
                 << "_" << artifact.stem << ".png";
        const auto output = examples_directory / filename.str();
        writeImage(output, makeExampleSheet(artifact));
        const std::string reason = artifact.status == "good"
            ? "sequence_sample" : "needs_review";
        metadata << i + 1 << ',' << artifact.pair_index
                 << ',' << artifact.ccd_index
                 << ',' << artifact.xray_index
                 << ',' << artifact.status << ',' << artifact.iou
                 << ',' << reason << ',' << csvText(filename.str()) << '\n';
    }
    metadata.close();
    if (!metadata) throw std::runtime_error("Cannot finish examples metadata");
    return selected.size();
}

} // namespace

SequenceAlignment associateObjectSequences(
    const std::vector<RegistrationObject>& ccd,
    const std::vector<RegistrationObject>& xray,
    const RegistrationOptions& options
) {
    if (ccd.empty() || xray.empty()) throw std::invalid_argument("Empty registration input");
    if (!(options.gap_cost > 0.0)) throw std::invalid_argument("gap_cost must be positive");
    const auto cf = featuresOf(ccd);
    const auto xf = featuresOf(xray);
    std::int64_t time_offset = estimateTimeOffset(ccd, xray, options);
    double anisotropy = distributionRatio(xf, cf, &ObjectFeature::aspect);
    double area_scale = distributionRatio(xf, cf, &ObjectFeature::area);

    SequenceAlignment result;
    for (int iteration = 0; iteration < 3; ++iteration) {
        result = dynamicAlignment(ccd, xray, cf, xf, time_offset,
                                  anisotropy, area_scale, options.gap_cost);
        if (result.matches.empty()) throw std::runtime_error("No sequence matches found");
        std::vector<std::int64_t> time_values;
        std::vector<double> aspect_values, area_values;
        for (const auto& pair : result.matches) {
            time_values.push_back(xray[pair.xray_position].timestamp_ms -
                                  ccd[pair.ccd_position].timestamp_ms);
            aspect_values.push_back(
                xf[pair.xray_position].aspect / cf[pair.ccd_position].aspect);
            area_values.push_back(
                xf[pair.xray_position].area / cf[pair.ccd_position].area);
        }
        time_offset = medianInt64(time_values);
        anisotropy = median(std::move(aspect_values));
        area_scale = median(std::move(area_values));
    }
    // Recompute once with the final robust estimates so the mapping and the
    // association costs written to CSV describe the same model.
    return dynamicAlignment(ccd, xray, cf, xf, time_offset,
                            anisotropy, area_scale, options.gap_cost);
}

RegistrationSummary registerExportedRun(
    const std::filesystem::path& results_directory,
    const std::string& requested_run,
    const RegistrationOptions& options,
    const VisualizationOptions& visualization
) {
    if (!std::filesystem::is_directory(results_directory))
        throw std::runtime_error("Results directory does not exist: " +
                                 results_directory.string());
    if (!(options.scale_search_step > 0.0) ||
        !(options.scale_search_fraction >= 0.0) ||
        !(options.rotation_step_deg > 0.0) ||
        !(options.max_rotation_deg >= 0.0) ||
        options.residual_shift_pixels < 0) {
        throw std::invalid_argument("Invalid registration search options");
    }
    if (!std::isfinite(visualization.overlay_alpha) ||
        visualization.overlay_alpha < 0.0 || visualization.overlay_alpha > 1.0) {
        throw std::invalid_argument("overlay_alpha must be in [0, 1]");
    }
    colormapCode(visualization.colormap);
    const std::string run = selectRun(results_directory, requested_run);
    const auto ccd_root = results_directory / "ccd" / run;
    const auto xray_root = results_directory / "x" / run;
    const auto ccd = loadObjects(ccd_root, true);
    const auto xray = loadObjects(xray_root, false);
    const auto cf = featuresOf(ccd);
    const auto xf = featuresOf(xray);
    const SequenceAlignment alignment = associateObjectSequences(ccd, xray, options);
    const auto calibrated = calibrateScaleAndRotation(
        ccd, xray, cf, xf, alignment, options);
    const double isotropic_scale = calibrated.first;
    const double rotation = calibrated.second;

    const auto final_root = uniqueOutputPath(results_directory / "registration", run);
    const auto temporary_root = std::filesystem::path(final_root.string() + ".partial");
    const auto registered_images = temporary_root / "registered_ccd" / "images";
    const auto registered_masks = temporary_root / "registered_ccd" / "masks";
    const auto registered_depth = temporary_root / "registered_ccd" / "depth";
    const auto paired_x_images = temporary_root / "x" / "images";
    const auto paired_x_masks = temporary_root / "x" / "masks";
    const auto overlays = temporary_root / "overlays";
    const auto previews = temporary_root / "previews";
    const auto transparent_fusions = temporary_root / "fusions" / "transparent";
    const auto false_color_fusions = temporary_root / "fusions" / "false_color";
    for (const auto& directory : {registered_images, registered_masks, registered_depth,
                                  paired_x_images, paired_x_masks, overlays, previews,
                                  transparent_fusions, false_color_fusions})
        std::filesystem::create_directories(directory);

    std::ofstream pairs_file(temporary_root / "pairs.csv");
    std::ofstream unmatched_file(temporary_root / "unmatched.csv");
    if (!pairs_file || !unmatched_file)
        throw std::runtime_error("Cannot create registration metadata");
    pairs_file << "pair_index,ccd_index,ccd_timestamp_ms,x_index,x_timestamp_ms,"
                  "time_delta_ms,time_residual_ms,association_cost,anisotropy_x_over_y,"
                  "isotropic_scale,scale_x,scale_y,rotation_deg,shift_x,shift_y,"
                  "center_iou,iou,dice,status,m00,m01,m02,m10,m11,m12,"
                  "registered_ccd_image,registered_ccd_mask,registered_ccd_depth,"
                  "x_image,x_mask,overlay,preview,transparent_fusion,"
                  "false_color_fusion\n";
    unmatched_file << "side,index,timestamp_ms,image_path,mask_path\n";
    pairs_file << std::setprecision(17);

    std::vector<double> ious;
    std::vector<PairArtifact> artifacts;
    artifacts.reserve(alignment.matches.size());
    std::size_t review_count = 0, low_count = 0;
    for (std::size_t p = 0; p < alignment.matches.size(); ++p) {
        const auto& sequence = alignment.matches[p];
        const auto& c = ccd[sequence.ccd_position];
        const auto& x = xray[sequence.xray_position];
        const auto centered_matrix = transformMatrix(
            cf[sequence.ccd_position], xf[sequence.xray_position],
            alignment.anisotropy_x_over_y, isotropic_scale, rotation);
        cv::Mat best_mask = warpMask(c.mask, centered_matrix, x.mask.size());
        const double center_iou = overlapScores(best_mask, x.mask).first;
        double best_iou = center_iou;
        int best_dx = 0, best_dy = 0;
        cv::Matx23d best_matrix = centered_matrix;
        for (int dy = -options.residual_shift_pixels;
             dy <= options.residual_shift_pixels; ++dy) {
            for (int dx = -options.residual_shift_pixels;
                 dx <= options.residual_shift_pixels; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const auto candidate_matrix = transformMatrix(
                    cf[sequence.ccd_position], xf[sequence.xray_position],
                    alignment.anisotropy_x_over_y, isotropic_scale, rotation, dx, dy);
                cv::Mat candidate = warpMask(c.mask, candidate_matrix, x.mask.size());
                const double candidate_iou = overlapScores(candidate, x.mask).first;
                if (candidate_iou > best_iou) {
                    best_iou = candidate_iou;
                    best_dx = dx;
                    best_dy = dy;
                    best_matrix = candidate_matrix;
                    best_mask = std::move(candidate);
                }
            }
        }
        const double dice = overlapScores(best_mask, x.mask).second;
        std::string status = "good";
        if (best_iou < options.low_iou) {
            status = "low_score";
            ++low_count;
        } else if (best_iou < options.review_iou) {
            status = "review";
            ++review_count;
        }
        ious.push_back(best_iou);

        cv::Mat ccd_image = readImage(c.image_path, cv::IMREAD_UNCHANGED);
        cv::Mat ccd_depth = readImage(c.depth_path, cv::IMREAD_UNCHANGED);
        cv::Mat xray_image = readImage(x.image_path, cv::IMREAD_UNCHANGED);
        if (ccd_image.type() != CV_8UC1 || ccd_depth.type() != CV_16SC1 ||
            xray_image.type() != CV_16UC1)
            throw std::runtime_error("Unexpected exported image type");
        cv::Mat warped_image, warped_depth;
        cv::warpAffine(ccd_image, warped_image, cv::Mat(best_matrix), x.mask.size(),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::warpAffine(ccd_depth, warped_depth, cv::Mat(best_matrix), x.mask.size(),
                       cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(-32768));
        warped_image.setTo(0, best_mask == 0);
        warped_depth.setTo(-32768, best_mask == 0);

        const std::string name = pairStem(p + 1, c.index, x.index);
        const auto registered_image_path = registered_images / (name + ".png");
        const auto registered_mask_path = registered_masks / (name + ".png");
        const auto registered_depth_path = registered_depth / (name + ".tiff");
        const auto x_image_path = paired_x_images / (name + ".png");
        const auto x_mask_path = paired_x_masks / (name + ".png");
        const auto overlay_path = overlays / (name + ".png");
        const auto preview_path = previews / (name + ".png");
        const auto transparent_path = transparent_fusions / (name + ".png");
        const auto false_color_path = false_color_fusions / (name + ".png");
        writeImage(registered_image_path, warped_image);
        writeImage(registered_mask_path, best_mask);
        writeImage(registered_depth_path, warped_depth);
        writeImage(x_image_path, xray_image);
        writeImage(x_mask_path, x.mask);
        const cv::Mat overlay = agreementOverlay(xray_image, best_mask, x.mask);
        const cv::Mat transparent = transparentFusion(
            xray_image, x.mask, warped_image, best_mask, visualization);
        const cv::Mat false_color = falseColorFusion(
            xray_image, x.mask, warped_image, best_mask);
        writeImage(overlay_path, overlay);
        writeImage(transparent_path, transparent);
        writeImage(false_color_path, false_color);
        writeImage(preview_path, makePreview(
            warped_image, xray_image, overlay, status, best_iou));
        artifacts.push_back(PairArtifact{
            p + 1, c.index, x.index, best_iou, status, name,
            c.image_path, x.image_path, registered_image_path,
            transparent_path, false_color_path});

        const auto relative = [&](const std::filesystem::path& path) {
            return std::filesystem::relative(path, temporary_root).generic_string();
        };
        const std::int64_t delta = x.timestamp_ms - c.timestamp_ms;
        pairs_file << p + 1 << ',' << c.index << ',' << c.timestamp_ms
                   << ',' << x.index << ',' << x.timestamp_ms
                   << ',' << delta << ',' << delta - alignment.time_offset_ms
                   << ',' << sequence.association_cost
                   << ',' << alignment.anisotropy_x_over_y
                   << ',' << isotropic_scale
                   << ',' << alignment.anisotropy_x_over_y * isotropic_scale
                   << ',' << isotropic_scale << ',' << rotation
                   << ',' << best_dx << ',' << best_dy
                   << ',' << center_iou << ',' << best_iou << ',' << dice
                   << ',' << status
                   << ',' << best_matrix(0, 0) << ',' << best_matrix(0, 1)
                   << ',' << best_matrix(0, 2) << ',' << best_matrix(1, 0)
                   << ',' << best_matrix(1, 1) << ',' << best_matrix(1, 2)
                   << ',' << csvText(relative(registered_image_path))
                   << ',' << csvText(relative(registered_mask_path))
                   << ',' << csvText(relative(registered_depth_path))
                   << ',' << csvText(relative(x_image_path))
                   << ',' << csvText(relative(x_mask_path))
                   << ',' << csvText(relative(overlay_path))
                   << ',' << csvText(relative(preview_path))
                   << ',' << csvText(relative(transparent_path))
                   << ',' << csvText(relative(false_color_path)) << '\n';
    }

    for (const auto position : alignment.unmatched_ccd_positions) {
        const auto& object = ccd[position];
        unmatched_file << "ccd," << object.index << ',' << object.timestamp_ms
                       << ',' << csvText(object.image_path.generic_string())
                       << ',' << csvText(object.mask_path.generic_string()) << '\n';
    }
    for (const auto position : alignment.unmatched_xray_positions) {
        const auto& object = xray[position];
        unmatched_file << "x," << object.index << ',' << object.timestamp_ms
                       << ',' << csvText(object.image_path.generic_string())
                       << ',' << csvText(object.mask_path.generic_string()) << '\n';
    }
    pairs_file.close();
    unmatched_file.close();
    if (!pairs_file || !unmatched_file)
        throw std::runtime_error("Cannot finish registration metadata");

    const std::size_t example_count = writeExamples(
        temporary_root, artifacts, visualization);

    RegistrationSummary summary;
    summary.source_run = run;
    summary.output_directory = final_root;
    summary.ccd_count = ccd.size();
    summary.xray_count = xray.size();
    summary.matched_count = alignment.matches.size();
    summary.unmatched_ccd_count = alignment.unmatched_ccd_positions.size();
    summary.unmatched_xray_count = alignment.unmatched_xray_positions.size();
    summary.review_count = review_count;
    summary.low_score_count = low_count;
    summary.example_count = example_count;
    summary.time_offset_ms = alignment.time_offset_ms;
    summary.anisotropy_x_over_y = alignment.anisotropy_x_over_y;
    summary.isotropic_scale = isotropic_scale;
    summary.scale_x = alignment.anisotropy_x_over_y * isotropic_scale;
    summary.scale_y = isotropic_scale;
    summary.rotation_deg = rotation;
    summary.median_iou = median(ious);

    std::ofstream report(temporary_root / "summary.txt");
    report << std::setprecision(12)
           << "source_run=" << run << '\n'
           << "ccd_count=" << summary.ccd_count << '\n'
           << "xray_count=" << summary.xray_count << '\n'
           << "matched_count=" << summary.matched_count << '\n'
           << "unmatched_ccd_count=" << summary.unmatched_ccd_count << '\n'
           << "unmatched_xray_count=" << summary.unmatched_xray_count << '\n'
           << "time_offset_ms=" << summary.time_offset_ms << '\n'
           << "anisotropy_x_over_y=" << summary.anisotropy_x_over_y << '\n'
           << "isotropic_scale=" << summary.isotropic_scale << '\n'
           << "scale_x=" << summary.scale_x << '\n'
           << "scale_y=" << summary.scale_y << '\n'
           << "rotation_deg=" << summary.rotation_deg << '\n'
           << "median_iou=" << summary.median_iou << '\n'
           << "good_count=" << summary.matched_count - review_count - low_count << '\n'
           << "review_count=" << review_count << '\n'
           << "low_score_count=" << low_count << '\n'
           << "transparent_overlay_alpha=" << visualization.overlay_alpha << '\n'
           << "pseudocolor_map=" << visualization.colormap << '\n'
           << "example_count=" << example_count << '\n'
           << "overlay_colors=yellow:overlap,green:ccd_only,red:x_only\n"
           << "note=Object centers are aligned first; one global camera scale and rotation are used.\n";
    report.close();
    if (!report) throw std::runtime_error("Cannot finish registration summary");
    std::ofstream complete(temporary_root / "COMPLETE.txt");
    complete << "Registration completed. Review low_score and review rows in pairs.csv.\n";
    complete.close();
    if (!complete) throw std::runtime_error("Cannot mark registration complete");

    std::filesystem::create_directories(final_root.parent_path());
    std::filesystem::rename(temporary_root, final_root);
    return summary;
}

} // namespace object_registration
