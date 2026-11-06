#include "registration/object_registration.hpp"
#include "registration/registration_config.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: register_objects <registration_config_or_results_directory> [run_name]\n";
        return 1;
    }
    try {
        const std::filesystem::path input = std::filesystem::u8path(argv[1]);
        const std::string run = argc == 3 ? argv[2] : std::string();
        object_registration::RegistrationOptions registration;
        object_registration::VisualizationOptions visualization;
        std::filesystem::path results;
        if (std::filesystem::is_directory(input)) {
            results = input;
        } else {
            const auto config = object_registration::loadRegistrationConfig(input);
            results = config.results_directory;
            registration = config.registration;
            visualization = config.visualization;
        }
        const auto summary = object_registration::registerExportedRun(
            results, run, registration, visualization);
        std::cout << "Source run: " << summary.source_run
                  << "\nCCD/X: " << summary.ccd_count << '/' << summary.xray_count
                  << "\nMatched: " << summary.matched_count
                  << "\nUnmatched CCD/X: " << summary.unmatched_ccd_count
                  << '/' << summary.unmatched_xray_count
                  << "\nTime offset: " << summary.time_offset_ms << " ms"
                  << "\nScale x/y: " << summary.scale_x << '/' << summary.scale_y
                  << "\nRotation: " << summary.rotation_deg << " deg"
                  << "\nMedian IoU: " << summary.median_iou
                  << "\nReview/low score: " << summary.review_count
                  << '/' << summary.low_score_count
                  << "\nTypical examples: " << summary.example_count
                  << "\nOutput: " << summary.output_directory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Registration failed: " << error.what() << '\n';
        return 1;
    }
}
