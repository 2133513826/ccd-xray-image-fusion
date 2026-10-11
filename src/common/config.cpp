#include "common/config.hpp"
#include<opencv2/core.hpp>
#include<fstream>
#include<stdexcept>
#include<cmath>

namespace fs = std::filesystem;

AppConfig loadConfig(const fs::path& file){
    //将yaml文件读取为字符串
    std::ifstream input(file);
    if(!input.is_open()){
        throw std::runtime_error("Failed to open config file: "+file.string());
    }
    std::string text(std::istreambuf_iterator<char>(input),{});
    cv::FileStorage yaml(text,cv::FileStorage::READ|cv::FileStorage::MEMORY);
    if(!yaml.isOpened()){
        throw std::runtime_error("Failed to parse config file: "+file.string());
    }
    //读取path节点
    auto paths=yaml["paths"];
    if(!paths.isMap()){
        throw std::runtime_error("Missing 'path' mapping in config file: "+file.string());
    }
    //读取path参数
    auto resolve=[&](const char* key)
    {
     auto node=paths[key];
     if(!node.isString()){
        throw std::runtime_error(std::string("Missing or invalid '")+key+"' in config file: "+file.string());
     }
     
     std::string path_str=node.string();
     if(path_str.empty()){
        throw std::runtime_error(std::string("Empty '")+key+"' in config file: "+file.string());
     }

     auto p=fs::u8path(path_str);
     return (p.is_absolute()?p:fs::absolute(file).parent_path()/p).lexically_normal();
    };
     //读取ccdoptions
     auto ccd_options=yaml["ccd_preprocess"];
     if(!ccd_options.isMap()){
        throw std::runtime_error("Missing'ccd_preprocess'mapping in config file:"+file.string());
     }

    //读取ccd_dir和output_dir
    AppConfig cfg;
    cfg.ccd_dir=resolve("ccd_dir");
    cfg.output_dir=resolve("output_dir");

    if(!fs::is_directory(cfg.ccd_dir)){
        throw std::runtime_error("CCD directory does not exist: "+cfg.ccd_dir.string());
    }
    fs::create_directories(cfg.output_dir);




ccd_options["invalid_depth"]
    >> cfg.ccd_options.invalid_depth;

ccd_options["valid_ratio"]
    >> cfg.ccd_options.valid_ratio;

ccd_options["depth_diff"]
    >> cfg.ccd_options.depth_diff;

ccd_options["intensity_diff"]
    >> cfg.ccd_options.intensity_diff;

ccd_options["morph_kernel"]
    >> cfg.ccd_options.kernel;

ccd_options["min_area"]
    >> cfg.ccd_options.min_area;
    ccd_options["edge_margin"] >> cfg.ccd_options.edge_margin;
ccd_options["padding_pixels"] >> cfg.ccd_options.padding_pixels;


// 使用已有resolve函数，使相对地址以配置文件所在目录为基准。
cfg.x_high_dir = resolve("x_high_dir");
if (!fs::is_directory(cfg.x_high_dir))
    throw std::runtime_error("X-ray HIGH directory does not exist");

// 读取X光参数节点。
const auto x = yaml["x_preprocess"];
if (!x.isMap())
    throw std::runtime_error("Missing x_preprocess mapping");

// 检查每个参数必须存在且为整数；整数0也是合法输入。
auto readInt = [&](const char* key, int& value) {
    const auto node = x[key];
    if (!node.isInt())
        throw std::runtime_error(std::string("Missing or invalid X-ray option: ") + key);
    node >> value;
};

auto& p = cfg.x_options;
readInt("frame_interval_ms", p.frame_interval_ms);
readInt("time_tolerance_ms", p.time_tolerance_ms);
readInt("foreground_threshold", p.foreground_threshold);
readInt("morph_kernel", p.morph_kernel);
readInt("min_area", p.min_area);
readInt("edge_margin", p.edge_margin);
readInt("padding_pixels", p.padding_pixels);
readInt("valid_x_begin", p.valid_x_begin);
readInt("valid_x_end", p.valid_x_end);

// 该参数可以是小数，不能使用只接受整数的readInt。
const auto node = x["min_horizontal_iou"];
if (!node.isInt() && !node.isReal()) {
    throw std::runtime_error("Invalid min_horizontal_iou");
}
node >> p.min_horizontal_iou;

// 检查数值范围，避免错误配置进入关联算法。
if (!std::isfinite(p.min_horizontal_iou) ||
    p.min_horizontal_iou <= 0.0 ||
    p.min_horizontal_iou > 1.0) {
    throw std::runtime_error("min_horizontal_iou must be in (0, 1]");
}
readInt("core_threshold", p.core_threshold);
readInt("min_core_pixels", p.min_core_pixels);

    return cfg;
}