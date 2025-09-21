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



#---------------------------------------------------------------------------------------------------------------
#NOTE: below allows a basic YOLOv5s version to find objects that it was pretrained on!!
#cmake_minimum_required(VERSION 3.16)
#project(FinalProject)
#set(CMAKE_CXX_STANDARD 17)
#
## Your working path (MinGW build):
#set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build/x64/mingw/lib")
#
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_executable(FinalProject
#        main.cpp
#        extraFunctions.cpp
#        Player.cpp)
#
#target_link_libraries(FinalProject ${OpenCV_LIBS})
#
## (Optional) sanity print
#message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
#
## Copy assets (models, labels, test images) next to the exe after build:
#add_custom_command(TARGET FinalProject POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_directory
#        "${CMAKE_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:FinalProject>/assets")
#NOTE: ABOVE allows a basic YOLOv5s version to find objects that it was pretrained on!!
#----------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
#NOTE: below is experimental and trying to get the weights loaded into my c++ project from when i trained on the
#filled_board_no_arms annotated images

cmake_minimum_required(VERSION 3.16)
project(FinalProject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Points to the folder that contains OpenCVConfig.cmake
set(OpenCV_DIR "C:/OpenCVforCplusplus/opencv/build")

# ✅ Option A: request all needed components
find_package(OpenCV REQUIRED COMPONENTS
        core imgproc highgui imgcodecs videoio dnn
        features2d calib3d flann
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

add_executable(FinalProject
        main.cpp
        extraFunctions.cpp
        Player.cpp
)

target_include_directories(FinalProject PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FinalProject PRIVATE ${OpenCV_LIBS})

if (WIN32)
    target_compile_definitions(FinalProject PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

add_custom_command(TARGET FinalProject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:FinalProject>/assets")

if (WIN32 AND DEFINED OpenCV_DIR)
    set(_opencv_bins
            "${OpenCV_DIR}/x64/mingw/bin"
            "${OpenCV_DIR}/bin"
            "${OpenCV_DIR}/x64/mingw/lib")
    foreach(_b ${_opencv_bins})
        if (EXISTS "${_b}")
            add_custom_command(TARGET FinalProject POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_b}" "$<TARGET_FILE_DIR:FinalProject>"
                    COMMENT "Copying OpenCV runtime from ${_b}")
            break()
        endif()
    endforeach()
endif()










