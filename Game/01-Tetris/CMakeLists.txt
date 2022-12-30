cmake_minimum_required(VERSION 3.23)
project(01_Tetris)

set(CMAKE_CXX_STANDARD 14)
link_libraries(libeasyx.a)
add_executable(01_Tetris main.cpp Block.h Tetris.h Block.cpp Tetris.cpp)
