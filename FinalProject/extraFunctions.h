//
// Created by RyanA on 2/22/2025.
//

#ifndef EXTRAFUNCTIONS_H
#define EXTRAFUNCTIONS_H

#endif //EXTRAFUNCTIONS_H
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;


void findCameraDetails();
tuple<cv::Mat, cv::Mat> findIntrinsicCameraMatrices();
void testVideoWithUndistortingEachFrame(int CAMERA_INDEX, cv::Mat camera_matrix, cv::Mat dist_coeffs);
void takeMultiplePictures(int CAMERA_INDEX, string imageName, int numImages);
void takeASinglePicture(int CAMERA_INDEX, string imagePathANDnameANDextension);
void takeASingleVideo(int CAMERA_INDEX);
void gettingSingleFrameFromAngledVideo();
void measureFPS(int CAMERA_INDEX);
void simpleYOLOv5Detection(cv::Mat img);
static std::vector<std::string> loadNames(const std::string& p);


