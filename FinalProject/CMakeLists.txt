cmake_minimum_required(VERSION 3.16)
project(FinalProject)
set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FinalProject main.cpp
        extraFunctions.cpp
        Player.cpp)
target_link_libraries(FinalProject ${OpenCV_LIBS})
