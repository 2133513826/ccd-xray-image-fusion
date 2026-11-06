#pragma once

#include "registration/object_registration.hpp"

#include <filesystem>

namespace object_registration {

struct RegistrationConfig {
    std::filesystem::path results_directory;
    RegistrationOptions registration;
    VisualizationOptions visualization;
};

RegistrationConfig loadRegistrationConfig(
    const std::filesystem::path& config_file
);

} // namespace object_registration
