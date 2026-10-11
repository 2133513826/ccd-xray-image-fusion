cmake_minimum_required(VERSION 3.20)

project(firstyear_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(image_fusion
    src/main.cpp
    src/io/data_loader.cpp
    src/common/config.cpp
    src/common/progress.cpp
    src/ccd/ccd_preprocess.cpp
    src/ccd/ccd_allmediaprocess.cpp
    src/ccd/ccd_crop.cpp
    src/ccd/ccd_sequence.cpp
    src/ccd/ccd_temporal.cpp
    src/x/x_objects.cpp
)

target_include_directories(image_fusion PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image_fusion PRIVATE
    ${OpenCV_LIBS}
)
