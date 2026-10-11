#include "common/progress.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

void showProgress(
    const std::string& stage,
    std::size_t completed,
    std::size_t total
) {
    if (total == 0) return;
    completed = std::min(completed, total);

    const double ratio = static_cast<double>(completed) / total;
    const int percent = static_cast<int>(ratio * 100);

    // Refresh only when the integer percentage changes, or at either endpoint.
    if (completed > 0 && completed < total) {
        const int previous_percent = static_cast<int>(
            static_cast<double>(completed - 1) / total * 100
        );
        if (percent == previous_percent) return;
    }

    constexpr int width = 25;
    const int filled = static_cast<int>(ratio * width);
    std::cout << '\r' << stage << " ["
              << std::string(filled, '#')
              << std::string(width - filled, '-')
              << "] " << std::setw(3) << percent << "% ("
              << completed << "/" << total << ")   " << std::flush;

    if (completed == total) std::cout << '\n';
}
