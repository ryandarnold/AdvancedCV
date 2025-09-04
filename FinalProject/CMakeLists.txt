#cmake_minimum_required(VERSION 3.16)
#project(FinalProject)
#set(CMAKE_CXX_STANDARD 17)
#
#set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")
#
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_executable(FinalProject main.cpp
#        extraFunctions.cpp
#        Player.cpp)
#target_link_libraries(FinalProject ${OpenCV_LIBS})

#NOTE: RYAN: THE ABOVE IS WORKING FOR THE ORIGINAL ADVANCED CV VERSION!!!



cmake_minimum_required(VERSION 3.16)
project(FinalProject)
set(CMAKE_CXX_STANDARD 17)

# Your working path (MinGW build):
set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FinalProject
        main.cpp
        extraFunctions.cpp
        Player.cpp)

target_link_libraries(FinalProject ${OpenCV_LIBS})

# (Optional) sanity print
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Copy assets (models, labels, test images) next to the exe after build:
add_custom_command(TARGET FinalProject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:FinalProject>/assets")


