cmake_minimum_required(VERSION 3.20)

project(firstyear_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(object_registration_core
    src/registration/object_registration.cpp
    src/registration/registration_config.cpp
)
target_include_directories(object_registration_core PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(object_registration_core PUBLIC
    ${OpenCV_LIBS}
)

add_executable(register_objects
    src/registration_main.cpp
)
target_link_libraries(register_objects PRIVATE object_registration_core)

add_executable(image_fusion
    src/main.cpp
    src/io/data_loader.cpp
    src/io/result_export.cpp
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

include(CTest)
if(BUILD_TESTING)
    add_executable(ccd_temporal_test
        tests/ccd_temporal_test.cpp
        src/ccd/ccd_preprocess.cpp
        src/ccd/ccd_sequence.cpp
        src/ccd/ccd_crop.cpp
        src/ccd/ccd_temporal.cpp
        src/common/progress.cpp
    )
    target_include_directories(ccd_temporal_test PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ccd_temporal_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME ccd_temporal_recovery COMMAND ccd_temporal_test)
    add_executable(result_export_test tests/result_export_test.cpp src/io/result_export.cpp)
    target_include_directories(result_export_test PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(result_export_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME result_export_roundtrip COMMAND result_export_test)
    add_executable(object_registration_test tests/object_registration_test.cpp)
    target_link_libraries(object_registration_test PRIVATE object_registration_core)
    add_test(NAME object_registration_sequence COMMAND object_registration_test)
endif()
