cmake_minimum_required(VERSION 3.16)
project(ORBSLAM3_Python)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ✅ Find OpenGL
find_package(OpenGL REQUIRED)

# ✅ Alternative GLEW finding methods
find_path(GLEW_INCLUDE_DIR GL/glew.h)
find_library(GLEW_LIBRARY NAMES GLEW glew32 glew)

if(GLEW_INCLUDE_DIR AND GLEW_LIBRARY)
    set(GLEW_FOUND TRUE)
    set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIR})
    set(GLEW_LIBRARIES ${GLEW_LIBRARY})
else()
    # Try using find_package for GLEW
    find_package(GLEW)
endif()

if(NOT GLEW_FOUND)
    message(WARNING "GLEW not found, trying to continue without it")
    set(GLEW_INCLUDE_DIRS "")
    set(GLEW_LIBRARIES "")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED)

pybind11_add_module(orbslam3 orbslam_pybind.cpp)

target_include_directories(orbslam3 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/include/CameraModels
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/Sophus
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/g2o/g2o
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/DBoW2/include
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_directories(orbslam3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/lib)

target_link_libraries(orbslam3 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM3/lib/libORB_SLAM3.so  # ← CHANGED LINE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OpenGL_LIBRARIES}
    ${GLEW_LIBRARIES}
)

if(GLEW_FOUND)
    target_compile_definitions(orbslam3 PRIVATE HAVE_GLEW)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(orbslam3 PRIVATE -O3 -DNDEBUG)
endif()

set_target_properties(orbslam3 PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_module
)