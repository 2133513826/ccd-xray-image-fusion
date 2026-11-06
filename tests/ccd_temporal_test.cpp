#include "ccd/ccd_temporal.hpp"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

CcdOptions options() {
    return {-32768.0, .8, 12.0, 20.0, 3, 200, 1, 1};
}

CcdBackground background(cv::Size size) {
    return {cv::Mat::zeros(size, CV_64F),
            cv::Mat(size, CV_64F, cv::Scalar(100)),
            cv::Mat(size, CV_8U, cv::Scalar(255))};
}

CcdPair makePair(const std::string& id, const std::vector<cv::Rect>& boxes) {
    CcdPair pair{id, cv::Mat::zeros(64, 96, CV_16S),
                 cv::Mat(64, 96, CV_8U, cv::Scalar(100))};
    for (const auto& box : boxes) {
        pair.depth_image(box).setTo(100);
        pair.intensity_image(box).setTo(50);
    }
    return pair;
}

std::vector<CcdObjectCrop> objects(const std::vector<CcdPair>& pairs,
                                  const CcdBackground& b, const CcdOptions& p) {
    std::vector<CcdObjectCrop> result;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const auto mask = detectCcd(pairs[i].depth_image, pairs[i].intensity_image, b, p);
        const auto frame = analyzeCcdMask(i + 1, parseCcdTimestampMs(pairs[i].id), mask, p.edge_margin);
        const auto crops = cropCcdObjects(pairs[i], frame, mask, p.padding_pixels);
        result.insert(result.end(), crops.begin(), crops.end());
    }
    return result;
}

void syntheticTests() {
    const auto p = options();
    const auto b = background({96, 64});
    CcdTemporalOptions temporal;
    const std::string first = "20251127120000000", next = "20251127120000370";
    auto run = [&](const std::vector<CcdPair>& pairs) {
        return processCcdTemporalObjects(objects(pairs, b, p), pairs, b, p, temporal);
    };

    auto r = run({makePair(first, {{10, 20, 20, 20}})});
    check(r.final_ccd_objects.size() == 1 && r.merged_info.empty() &&
          r.reprocessed_frame_count == 0, "Interior target changed");

    r = run({makePair(first, {{10, 50, 20, 14}}), makePair(next, {{10, 0, 20, 14}})});
    check(r.final_ccd_objects.size() == 1 && r.merged_info.size() == 1 &&
          r.recovered_fragment_count == 0 && r.reprocessed_frame_count == 0,
          "Normal matching must not invoke recovery");

    r = run({makePair(first, {{10, 50, 20, 14}}), makePair(next, {{10, 0, 20, 4}})});
    check(r.final_ccd_objects.size() == 1 && r.recovered_fragment_count == 1 &&
          r.merged_info.front().match.current_recovered &&
          !r.merged_info.front().match.current_index, "Next small fragment was not recovered");

    r = run({makePair(first, {{10, 60, 20, 4}}), makePair(next, {{10, 0, 20, 14}})});
    check(r.final_ccd_objects.size() == 1 && r.recovered_fragment_count == 1 &&
          r.merged_info.front().match.previous_recovered &&
          !r.merged_info.front().match.previous_index, "Previous small fragment was not recovered");
    check(r.merged_info.front().match.time_gap_ms == 370, "Recovery time order is wrong");

    r = run({makePair(first, {{10, 50, 20, 14}, {40, 50, 20, 14}}),
             makePair(next, {{10, 0, 50, 3}})});
    check(r.recovered_fragment_count == 1 && r.reprocessed_frame_count == 1 &&
          r.final_ccd_objects.size() == 2 && r.unmatched_boundary_objects.size() == 1,
          "A recovered fragment was reused or a frame was processed twice");

    r = run({makePair(first, {{10, 50, 20, 14}}),
             makePair("20251127120000500", {{10, 0, 20, 4}})});
    check(r.reprocessed_frame_count == 0 && r.unmatched_boundary_objects.size() == 1,
          "Recovery ignored the time gap");

    r = run({makePair(first, {{10, 50, 20, 14}}), makePair(next, {{60, 0, 20, 4}})});
    check(r.recovered_fragment_count == 0 && r.unmatched_boundary_objects.size() == 1,
          "Recovery ignored the horizontal overlap");

    r = run({makePair(first, {{10, 0, 20, 14}})});
    check(r.unmatched_boundary_objects.size() == 1 && r.reprocessed_frame_count == 0,
          "First frame attempted backward access");
    r = run({makePair(first, {{10, 50, 20, 14}})});
    check(r.unmatched_boundary_objects.size() == 1 && r.reprocessed_frame_count == 0,
          "Last frame attempted forward access");

    r = run({makePair(first, {{10, 60, 20, 4}}), makePair(next, {{10, 0, 20, 4}})});
    check(r.final_ccd_objects.empty(), "Two filtered halves should not create an anchor");
    r = run({});
    check(r.final_ccd_objects.empty(), "Empty input failed");

    auto noise = makePair(first, {});
    noise.intensity_image.at<uchar>(20, 20) = 0;
    auto relaxed = p;
    relaxed.min_area = 1;
    check(cv::countNonZero(detectCcd(noise.depth_image, noise.intensity_image, b, relaxed)) == 0,
          "Opening did not remove isolated noise");

    r = run({makePair(first, {{10, 50, 20, 14}}), makePair(next, {{15, 0, 3, 64}})});
    check(r.recovered_fragment_count == 0, "Both-boundary noise should not be recovered");

    check(parseCcdTimestampMs(next) - parseCcdTimestampMs(first) == 370,
          "Timestamp parsing failed");
    bool rejected = false;
    try { parseCcdTimestampMs("20250230120000000"); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid calendar date was accepted");
    std::cout << "Synthetic CCD recovery tests passed\n";
}

void realSampleTests(const std::filesystem::path& folder) {
    auto p = options();
    p.min_area = 1000; p.edge_margin = 3; p.padding_pixels = 5;
    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        const auto name = entry.path().filename().string();
        if (name.size() == 28 && name.substr(0, 6) == "Image_" &&
            name.substr(23) == ".tiff") ids.push_back(name.substr(6, 17));
    }
    std::sort(ids.begin(), ids.end());
    auto load = [&](const std::string& id) {
        CcdPair pair{id,
            cv::imread((folder / ("Image_" + id + ".tiff")).string(), cv::IMREAD_UNCHANGED),
            cv::imread((folder / ("IntensityImageImage_" + id + ".jpg")).string(), cv::IMREAD_GRAYSCALE)};
        check(!pair.depth_image.empty() && !pair.intensity_image.empty(), "Sample image is missing");
        return pair;
    };
    check(ids.size() >= 20, "Not enough background samples");
    std::vector<CcdPair> samples;
    for (int i = 0; i < 20; ++i) samples.push_back(load(ids[i]));
    const auto b = buildCcdBackground(samples, p);
    CcdTemporalOptions temporal;
    for (const auto& target : {"20251127155658595", "20251127155708949", "20251127155807740"}) {
        auto found = std::find(ids.begin(), ids.end(), target);
        check(found != ids.end() && found != ids.begin() && found + 1 != ids.end(), "Target frame is missing");
        std::vector<CcdPair> pairs{load(*(found - 1)), load(*found), load(*(found + 1))};
        const auto r = processCcdTemporalObjects(objects(pairs, b, p), pairs, b, p, temporal);
        const auto timestamp = parseCcdTimestampMs(target);
        bool recovered = false;
        for (const auto& merge : r.merged_info) {
            const auto& m = merge.match;
            if ((m.previous_info.timestamp_ms == timestamp || m.current_info.timestamp_ms == timestamp) &&
                (m.previous_recovered || m.current_recovered)) {
                recovered = true;
                std::cout << "Recovered sample " << target << " areas=" << m.previous_info.area
                          << "/" << m.current_info.area << " gap=" << m.time_gap_ms << '\n';
            }
        }
        check(recovered, "A real boundary sample was not recovered");
    }
    const auto noise = load("20251127155808106");
    const auto noise_mask = detectCcd(noise.depth_image, noise.intensity_image, b, p);
    std::cout << "Former belt-noise frame: remaining pixels=" << cv::countNonZero(noise_mask) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        syntheticTests();
        if (argc == 2) realSampleTests(argv[1]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CCD test failed: " << e.what() << '\n';
        return 1;
    }
}
