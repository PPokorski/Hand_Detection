cmake_minimum_required(VERSION 2.8)
project(pc_control)

set(OpenCV_DIR "/home/pokor/opencv_ws/opencv-2.4.9/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pc_control src/main.cpp)
target_link_libraries(pc_control ${OpenCV_LIBS})
