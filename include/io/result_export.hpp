#pragma once

#include "ccd/ccd_temporal.hpp"
#include "x/x_objects.hpp"
#include <filesystem>
#include <vector>

struct ObjectExportPaths {
    std::filesystem::path ccd;
    std::filesystem::path xray;
};

// Export original crop pixels, masks, depth and registration metadata.
ObjectExportPaths exportObjectResults(
    const std::filesystem::path& output_dir,
    const CcdTemporalResult& ccd,
    const std::vector<XrayObject>& xray
);
