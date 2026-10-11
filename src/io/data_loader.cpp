#include "io/data_loader.hpp"
#include "common/progress.hpp"
#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <stdexcept>
#include<iterator>
#include<utility>
#include<regex>

//读取一对ccd图像，返回是否成功
bool loadCcdPair(const std::string&depth_path,
     const std::string&intensity_path,
     cv::Mat& depth_image,
     cv::Mat& intensity_image){
depth_image = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
intensity_image = cv::imread(intensity_path, cv::IMREAD_GRAYSCALE);
   if (depth_image.empty()) {
    std::cerr << "Failed to read depth image: "
              << depth_path << std::endl;
return false;
                  }
   if(intensity_image.empty()) {
    std::cerr << "Failed to read intensity image: "
              << intensity_path << std::endl;
    return false;
   }

   if(depth_image.size() != intensity_image.size()) {
    std::cerr << "Depth and intensity images have different sizes: "
              << depth_path << " and " << intensity_path << std::endl;
    return false;
   }
   return true;
}

//读取数据并配对，返回所有配对的ccd信息
std::vector<CcdFilePair> inspectCcdFolder(const std::string& folder){
    namespace fs = std::filesystem;

    // 检查folder是否是目录
    if(!fs::is_directory(folder)){
        std::cerr<< "Provided path is not a directory: " << folder << std::endl;
        return {};
    }
    // 定义一张表来存放各组ccd的深度图和强度图路径
    struct PairPaths {
        std::string depth_path;
        std::string intensity_path;
    };
    std::map<std::string,PairPaths> ccd_map;

    // 定义命名规则
    const std::regex depth_regex(R"(^Image_([0-9]{17})\.tiff$)");
    const std::regex intensity_regex(R"(^IntensityImageImage_([0-9]{17})\.jpg$)");
    
    // 遍历目录下的所有图片，并归于ccd_map
    for(const auto& entry:fs::directory_iterator(folder)){
        // 只处理文件
        if(!entry.is_regular_file()) continue;
        
        const std::string filename = entry.path().filename().string();
     // 使用正则表达式匹配文件名,并存入ccd_map
        std::smatch match;
        if(std::regex_match(filename,match,depth_regex)){
            ccd_map[match[1].str()].depth_path=entry.path().string();
        }else if(std::regex_match(filename,match,intensity_regex)){
            ccd_map[match[1].str()].intensity_path=entry.path().string();
        }
    }

    std::vector<CcdFilePair> ccd_pairs;
    //检查每组ccd是否都有深度图和强度图
    for(const auto& [key,pair]: ccd_map) {
        if(pair.depth_path.empty()){
            std::cerr << "Missing depth image for CCD: " << key << std::endl;
        } else if(pair.intensity_path.empty()) {
            std::cerr << "Missing intensity image for CCD: " << key << std::endl;
        } else {
            ccd_pairs.push_back({key,pair.depth_path,pair.intensity_path});
        }
    }
std::cout<<"Complete CCD pairs found: "<<ccd_pairs.size()<<std::endl;
return ccd_pairs;
}

std::vector<CcdPair> loadCcdPairs(const std::vector<CcdFilePair>& file_pairs){
    std::vector<CcdPair> ccd_pairs;
    std::size_t completed = 0;
    showProgress("CCD read", 0, file_pairs.size());
    for(const auto& file_pair: file_pairs){
        cv::Mat depth_image, intensity_image;
        if(loadCcdPair(file_pair.depth_path,file_pair.intensity_path,depth_image,intensity_image)){
            ccd_pairs.push_back({file_pair.id,depth_image,intensity_image});
        }else{
            std::cerr<<"Failed to load CCD pair: "<<file_pair.id<<std::endl;
        }
        showProgress("CCD read", ++completed, file_pairs.size());
    }
    return ccd_pairs;
}

// 生成X射线掩膜,完成去噪与小物体过滤，皮带边界处理
cv::Mat generateXrayMask(
    const cv::Mat& high,
    const XrayOptions& p
){ // 保留16位原始灰度进行判断。
    if (high.empty() || high.type() != CV_16UC1)
        throw std::invalid_argument("HIGH must be nonempty CV_16UC1");

        // 高阈值必须高于低阈值，并且不能超出16位灰度范围。
if (p.core_threshold <= p.foreground_threshold ||
    p.core_threshold > 65535 ||
    p.min_core_pixels <= 0) {
    throw std::invalid_argument("Invalid X-ray core parameters");
}

    // 阈值允许为0；形态学核必须为正奇数。
    if (p.foreground_threshold < 0 || p.foreground_threshold > 65535 ||
        p.morph_kernel <= 0 || p.morph_kernel % 2 == 0 || p.min_area <= 0)
        throw std::invalid_argument("Invalid X-ray mask parameters");


    // 1. 阈值分割，compare直接生成CV_8UC1的0/255掩膜。
    cv::Mat mask;
    cv::compare(high, p.foreground_threshold, mask, cv::CMP_GT);

    // 2. 排除配置范围之外的区域，保持原始图像坐标不变。
    cv::Mat valid_mask = cv::Mat::zeros(high.size(), CV_8UC1);
    // 有效横向范围为[valid_x_begin, valid_x_end)。
const int right =
    p.valid_x_end == -1 ? high.cols : p.valid_x_end;

if (p.valid_x_begin < 0 ||
    right > high.cols ||
    right <= p.valid_x_begin) {
    throw std::invalid_argument("Invalid X-ray detection range");
}

valid_mask(cv::Rect(
    p.valid_x_begin,
    0,
    right - p.valid_x_begin,
    high.rows
)).setTo(255);
    cv::bitwise_and(mask, valid_mask, mask);

    // 3. 开运算清除细小前景，闭运算连接小断裂、填补小孔洞。
    const auto kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(p.morph_kernel, p.morph_kernel)
    );
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 防止形态学处理后的前景进入禁用区域。
    cv::bitwise_and(mask, valid_mask, mask);

       // 4. 给候选mask的每个连通域分配标签。
    // 标签0代表背景，标签1、2……代表各个候选区域。
    cv::Mat labels, stats, centers;
    const int count = cv::connectedComponentsWithStats(
        mask, labels, stats, centers, 8, CV_32S
    );

    // core_counts[label]记录该区域内部的高灰度像素数。
    std::vector<int> core_counts(count, 0);

    for (int y = 0; y < high.rows; ++y) {
        // 原始HIGH为CV_16UC1，标签图为CV_32S。
        const auto* values = high.ptr<unsigned short>(y);
        const auto* row_labels = labels.ptr<int>(y);

        for (int x = 0; x < high.cols; ++x) {
            const int label = row_labels[x];

            // 只统计候选区域内部的亮像素，不统计背景。
            if (label > 0 && values[x] > p.core_threshold) {
                ++core_counts[label];
            }
        }
    }

    // 5. 记录哪些连通域通过筛选。
    // keep[0]保持为0，所以背景不会进入输出mask。
    std::vector<unsigned char> keep(count, 0);

    for (int label = 1; label < count; ++label) {
        const int area =
            stats.at<int>(label, cv::CC_STAT_AREA);

        // 面积和亮核心数量必须同时满足要求。
        if (area >= p.min_area &&
            core_counts[label] >= p.min_core_pixels) {
            keep[label] = 255;
        }
    }

    // 6. 根据标签生成最终mask。
    // 保留通过筛选的整个区域，包括它较暗的边缘。
    cv::Mat result = cv::Mat::zeros(high.size(), CV_8UC1);

    for (int y = 0; y < labels.rows; ++y) {
        const auto* row_labels = labels.ptr<int>(y);
        auto* output = result.ptr<unsigned char>(y);

        for (int x = 0; x < labels.cols; ++x) {
            output[x] = keep[row_labels[x]];
        }
    }

    return result;


}

XrayData loadXrayData(
    const std::filesystem::path& high_dir,
    const XrayOptions& options
)
    {
        namespace fs = std::filesystem; 
        // 通过文件流读取，避免OpenCV直接打开中文路径的问题。
    auto read=[](const fs::path& path) {
        std::ifstream input(path,std::ios::binary);
        if(!input) throw std::runtime_error("Cannot open X-ray image");
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
        cv::Mat image=cv::imdecode(bytes,cv::IMREAD_UNCHANGED);
        if(image.empty() || image.type()!=CV_16UC1)
            throw std::runtime_error("X-ray image must be 16-bit grayscale");
        return image;
    };

        const auto& p=options;
if(p.frame_interval_ms<=0 || p.time_tolerance_ms<0 ||
   p.time_tolerance_ms>=p.frame_interval_ms)
    throw std::invalid_argument("Invalid X-ray grouping parameters");
if(!fs::is_directory(high_dir))
    throw std::runtime_error("HIGH directory does not exist");
// 文件名是毫秒时间戳，map保证按时间排序。
std::map<std::int64_t,fs::path> files;
const std::regex pattern("[0-9]{13}");
for(const auto& entry:fs::directory_iterator(high_dir)) {
    if(!entry.is_regular_file() || entry.path().extension()!=".png") continue;
    const auto id=entry.path().stem().string();
    if(!std::regex_match(id,pattern))
        throw std::runtime_error("Invalid timestamp: "+id);
    if(!files.emplace(std::stoll(id),entry.path()).second)
        throw std::runtime_error("Duplicate timestamp: "+id);
}
if(files.empty()) throw std::runtime_error("No HIGH images found");
XrayData result;
std::size_t completed = 0;
showProgress("X read + mask", 0, files.size());
for(const auto& [time,path]:files) {
    cv::Mat image=read(path); // 保留原始16位灰度。
    if(!result.groups.empty() && image.size()!=result.groups.front().front().high.size())
        throw std::runtime_error("HIGH image sizes differ");
    bool new_group=result.groups.empty();
    if(!new_group) {
        const auto gap=time-result.groups.back().back().timestamp_ms;
        const std::int64_t interval=p.frame_interval_ms, tolerance=p.time_tolerance_ms;
        new_group=(gap<interval-tolerance || gap>interval+tolerance);
    }
    if(new_group) result.groups.emplace_back();
    // 路径、时间戳、HIGH、空mask，对应XrayFrame的四个字段。
    // 先生成mask，再移动图像；移动后不要继续使用image。
cv::Mat mask = generateXrayMask(image, p);
cv::Mat labels;

// mask要求为8位单通道图，0表示背景，非0表示前景。
const int total = cv::connectedComponents(
    mask, labels, 8, CV_32S
);

// 返回数量包含背景标签0，所以减去1。
const int object_count = total - 1;

// 原图、mask、时间戳与路径绑定在同一个XrayFrame中。
result.groups.back().push_back({
    path,
    time,
    std::move(image),
    std::move(mask),
    object_count
});
showProgress("X read + mask", ++completed, files.size());
}
return result;
    };









