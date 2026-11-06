#include "io/result_export.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

void validateImage(const cv::Mat& image, const cv::Mat& mask, int type) {
    if (image.empty() || mask.empty() || image.type() != type ||
        mask.type() != CV_8UC1 || image.size() != mask.size()) {
        throw std::invalid_argument("Invalid export image or mask");
    }
    cv::Mat invalid = (mask != 0) & (mask != 255);
    if (cv::countNonZero(invalid) != 0)
        throw std::invalid_argument("Export mask must contain only 0 and 255");
}

void writeImage(const std::filesystem::path& path, const cv::Mat& image) {
    // Encode first; filesystem streams also support non-ASCII Windows paths.
    std::vector<unsigned char> bytes;
    if (!cv::imencode(path.extension().string(), image, bytes))
        throw std::runtime_error("Cannot encode export image");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Cannot write image: " + path.string());
}

std::string stem(std::size_t index, std::int64_t timestamp) {
    std::ostringstream text;
    text << std::setfill('0') << std::setw(6) << index << '_' << timestamp;
    return text.str();
}

void rectangle(std::ostream& file, const cv::Rect& box) {
    file << ',' << box.x << ',' << box.y << ',' << box.width << ',' << box.height;
}

void createFolders(const std::filesystem::path& root, bool depth) {
    std::filesystem::create_directories(root / "images");
    std::filesystem::create_directories(root / "masks");
    if (depth) std::filesystem::create_directories(root / "depth");
}

} // namespace

ObjectExportPaths exportObjectResults(
    const std::filesystem::path& output_dir,
    const CcdTemporalResult& ccd,
    const std::vector<XrayObject>& xray
) {
    if (output_dir.empty()) throw std::invalid_argument("Empty export directory");
    for (const auto& object : ccd.final_ccd_objects) {
        validateImage(object.intensity, object.mask, CV_8UC1);
        if (object.depth.empty() || object.depth.size() != object.mask.size() ||
            object.depth.type() != CV_16SC1) {
            throw std::invalid_argument("CCD export depth must be CV_16SC1");
        }
    }
    for (const auto& object : xray) validateImage(object.high, object.mask, CV_16UC1);

    const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto run = "run_" + std::to_string(time);
    ObjectExportPaths paths{output_dir / "ccd" / run, output_dir / "x" / run};
    // Never mix a new result list with stale images from an earlier run.
    if (std::filesystem::exists(paths.ccd) || std::filesystem::exists(paths.xray))
        throw std::runtime_error("Export batch already exists");
    createFolders(paths.ccd, true);
    createFolders(paths.xray, false);

    std::ofstream cm(paths.ccd / "objects.csv");
    std::ofstream xm(paths.xray / "objects.csv");
    if (!cm || !xm) throw std::runtime_error("Cannot create export metadata");
    cm << "object_index,timestamp_ms,source_frame_index,source_label,area,box_x,box_y,box_width,box_height,crop_x,crop_y,crop_width,crop_height,coordinate_space,first_frame_index,first_timestamp_ms,last_frame_index,last_timestamp_ms,fusion_mode,overlap_rows,fusion_score,previous_recovered,current_recovered,unmatched_boundary,review_status,image_path,mask_path,depth_path\n";
    xm << "object_index,timestamp_ms,area,box_x,box_y,box_width,box_height,crop_x,crop_y,crop_width,crop_height,coordinate_space,review_status,image_path,mask_path\n";
    cm << std::setprecision(17);

    for (std::size_t i = 0; i < ccd.final_ccd_objects.size(); ++i) {
        const auto& object = ccd.final_ccd_objects[i];
        const auto& info = object.object_info;
        const auto name = stem(i + 1, info.timestamp_ms);
        writeImage(paths.ccd / "images" / (name + ".png"), object.intensity);
        writeImage(paths.ccd / "masks" / (name + ".png"), object.mask);
        writeImage(paths.ccd / "depth" / (name + ".tiff"), object.depth);

        const CcdObjectMerge* merge = nullptr;
        for (const auto& m : ccd.merged_info)
            if (m.merged_object.mask.data == object.mask.data) { merge = &m; break; }
        bool unmatched = false;
        for (const auto& u : ccd.unmatched_boundary_objects)
            if (u.mask.data == object.mask.data) unmatched = true;
        cm << i + 1 << ',' << info.timestamp_ms << ',' << info.file_index
           << ',' << info.label << ',' << cv::countNonZero(object.mask);
        rectangle(cm, info.bounding_rect);
        rectangle(cm, object.crop_box);
        // Existing merged boxes use source-image x, but merged-canvas y.
        cm << ',' << (merge ? "source_x_canvas_y" : "source_xy");
        const auto& first = merge ? merge->match.previous_info : info;
        const auto& last = merge ? merge->match.current_info : info;
        cm << ',' << first.file_index << ',' << first.timestamp_ms
           << ',' << last.file_index << ',' << last.timestamp_ms
           << ',' << (merge ? (merge->decision.mode == CcdFusionMode::DirectConcat
                              ? "direct_concat" : "redundancy_fusion") : "single_frame")
           << ',' << (merge && merge->decision.mode == CcdFusionMode::RedundancyFusion
                      ? merge->decision.best_overlap_rows : 0)
           << ',' << (merge ? merge->decision.best_score : 0.0)
           << ',' << (merge && merge->match.previous_recovered)
           << ',' << (merge && merge->match.current_recovered)
           << ',' << unmatched << ",unreviewed,images/" << name << ".png,masks/"
           << name << ".png,depth/" << name << ".tiff\n";
    }
    for (std::size_t i = 0; i < xray.size(); ++i) {
        const auto& object = xray[i];
        const auto name = stem(i + 1, object.timestamp_ms);
        writeImage(paths.xray / "images" / (name + ".png"), object.high);
        writeImage(paths.xray / "masks" / (name + ".png"), object.mask);
        xm << i + 1 << ',' << object.timestamp_ms << ',' << cv::countNonZero(object.mask);
        rectangle(xm, object.bounding_rect);
        rectangle(xm, object.crop_box);
        xm << ",source_xy,unreviewed,images/" << name << ".png,masks/" << name << ".png\n";
    }
    cm.close(); xm.close();
    if (!cm || !xm) throw std::runtime_error("Cannot finish export metadata");
    // Mark only fully written batches as complete.
    for (const auto& root : {paths.ccd, paths.xray}) {
        std::ofstream complete(root / "COMPLETE.txt");
        complete << "Export completed. Object indices are local to each camera, not matches.\n";
        complete.close();
        if (!complete) throw std::runtime_error("Cannot mark export complete");
    }
    return paths;
}
