cmake_minimum_required(VERSION 3.23)
project(Learn_OpenCV)

set(CMAKE_CXX_STANDARD 14)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "F:\\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\\")
find_package(OpenCV REQUIRED)
add_executable(Learn_OpenCV main.cpp quickopencv.h quickdemo.cpp)
# linking
target_link_libraries(Learn_OpenCV ${OpenCV_LIBS})

