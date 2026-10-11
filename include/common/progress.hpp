#pragma once

#include <cstddef>
#include <string>

// Print the completed work for one processing stage on a single line.
void showProgress(
    const std::string& stage,
    std::size_t completed,
    std::size_t total
);
