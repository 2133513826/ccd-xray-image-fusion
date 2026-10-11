#pragma once
#include <filesystem>
#include "ccd/ccd_preprocess.hpp"
#include "io/data_loader.hpp"

struct AppConfig {
    std::filesystem::path ccd_dir, output_dir;
    CcdOptions ccd_options;

    std::filesystem::path x_high_dir;
    XrayOptions x_options;
};

AppConfig loadConfig(const std::filesystem::path& file);