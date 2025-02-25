#cmake_minimum_required(VERSION 3.16)
#project(HW2)
#set(CMAKE_CXX_STANDARD 17)
#
#set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")
#
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_executable(HW2 main.cpp)
#target_link_libraries(HW2 ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.16)
project(HW2)
set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HW2 main.cpp)
target_link_libraries(HW2 ${OpenCV_LIBS})

