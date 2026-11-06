#include "io/result_export.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        auto root = std::filesystem::current_path() / "export_test_output" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        CcdObjectCrop c;
        c.object_info.file_index = 1; c.object_info.timestamp_ms = 123;
        c.object_info.bounding_rect = {10, 20, 2, 2}; c.crop_box = {10, 20, 2, 2};
        c.mask = (cv::Mat_<unsigned char>(2, 2) << static_cast<unsigned char>(0), 255, 255, 0);
        c.intensity = (cv::Mat_<unsigned char>(2, 2) << static_cast<unsigned char>(17), 128, 200, 255);
        c.depth = (cv::Mat_<short>(2, 2) << static_cast<short>(-32768), -5, 12, 32000);
        CcdTemporalResult ccd; ccd.final_ccd_objects.push_back(c);
        CcdObjectMerge merge;
        merge.merged_object = c;
        merge.merged_object.mask = c.mask.clone();
        merge.match.previous_info = c.object_info;
        merge.match.current_info = c.object_info;
        merge.match.current_info.file_index = 2;
        merge.match.current_info.timestamp_ms = 234;
        merge.match.previous_recovered = true;
        ccd.merged_info.push_back(merge);
        ccd.final_ccd_objects.push_back(merge.merged_object);
        XrayObject x; x.timestamp_ms = 456; x.mask = c.mask;
        x.high = (cv::Mat_<unsigned short>(2, 2) << static_cast<unsigned short>(0), 2501, 5000, 65535);
        auto paths = exportObjectResults(root, ccd, {x});
        for (const auto& item : std::vector<std::pair<std::filesystem::path, cv::Mat>>{
            {paths.ccd / "images/000001_123.png", c.intensity},
            {paths.ccd / "masks/000001_123.png", c.mask},
            {paths.ccd / "depth/000001_123.tiff", c.depth},
            {paths.xray / "images/000001_456.png", x.high},
            {paths.xray / "masks/000001_456.png", x.mask}}) {
            auto read = cv::imread(item.first.string(), cv::IMREAD_UNCHANGED);
            check(read.type() == item.second.type() && read.size() == item.second.size(), "Export type or size changed");
            check(cv::norm(read, item.second, cv::NORM_INF) == 0, "Export pixels changed");
        }
        std::ifstream csv(paths.ccd / "objects.csv");
        std::string header, row, merged_row; std::getline(csv, header); std::getline(csv, row);
        check(row.find("source_xy") != std::string::npos, "Missing coordinate metadata");
        std::getline(csv, merged_row);
        check(merged_row.find("source_x_canvas_y,1,123,2,234,direct_concat") != std::string::npos,
              "Missing merged-object provenance");
        csv.close();
        check(std::filesystem::exists(paths.ccd / "COMPLETE.txt"), "Missing completion marker");
        auto invalid = ccd; invalid.final_ccd_objects[0].mask = cv::Mat(2, 2, CV_8U, cv::Scalar(1));
        bool rejected = false;
        try { exportObjectResults(root, invalid, {x}); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Invalid mask accepted");
        // Only remove the uniquely created test directory, never user results.
        check(std::filesystem::weakly_canonical(root.parent_path()) ==
              std::filesystem::weakly_canonical(std::filesystem::current_path() / "export_test_output"),
              "Test cleanup path is outside its output directory");
        std::filesystem::remove_all(root);
        std::cout << "Export roundtrip passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
