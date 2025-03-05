#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "extraFunctions.h"
/*I have OpenCV version 4.5.5-dev
*/

using namespace std;

int CAMERA_INDEX = 0;

void display_image(cv::Mat original_image, double Scale, string window_name)
{
    cv::Mat resized_frame;
    cv::resize(original_image, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
    cv::waitKey(0);
}

cv::Mat testingSIFT(string board_template_name, string scene_image_name)
{
    // Load images
    cv::Mat board_img = cv::imread(board_template_name, cv::IMREAD_COLOR);
    cv::Mat scene_img = cv::imread(scene_image_name, cv::IMREAD_COLOR);

    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    sift->detectAndCompute(board_img, cv::noArray(), kp1, des1);
    sift->detectAndCompute(scene_img, cv::noArray(), kp2, des2);

    // Use FLANN-based matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);  // Find 2 nearest matches for each descriptor

    // Apply Lowe’s Ratio Test
    std::vector<cv::DMatch> good_matches;
    for (auto& m : knn_matches) {
        if (m[0].distance < 0.75 * m[1].distance) {  // Lowe's Ratio Test
            good_matches.push_back(m[0]);
        }
    }

    // Ensure enough good matches exist for homography
    if (good_matches.size() < 10) {
        //std::cout << "Error: Not enough good matches to compute homography!" << std::endl;
        throw std::invalid_argument("Error: Not enough good matches to compute homography!");
    }

    // Extract keypoint coordinates
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (auto& match : good_matches) {
        src_pts.push_back(kp1[match.queryIdx].pt); // Points in the original Monopoly board image
        dst_pts.push_back(kp2[match.trainIdx].pt); // Corresponding points in the second image
    }

    // Compute homography using RANSAC
    cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC);

    if (M.empty()) {
        // std::cout << "Error: Homography computation failed!" << std::endl;
        throw std::invalid_argument("Error: Homography computation failed!");
        // return;
    }

    // Warp the second image to align with the original board image
    cv::Mat aligned_scene;
    cv::warpPerspective(scene_img, aligned_scene, M, board_img.size());

    // Display results
    display_image(board_img, 0.5, "Original Monopoly Board");
    display_image(aligned_scene, 0.5, "aligned Monopoly Board");
    cv::destroyAllWindows();
    return aligned_scene;
}


cv::Mat crop_out_background(cv::Mat current_frame)
{
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(current_frame, gray, cv::COLOR_BGR2GRAY);

    // Apply thresholding to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);

    // Find contours of the image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour (assuming it's the Monopoly board)
    int largest_contour_index = -1;
    double max_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            largest_contour_index = i;
        }
    }

    if (largest_contour_index == -1) {
        // std::cout << "Error: Could not detect the board!" << std::endl;
        throw std::invalid_argument("Error: Could not detect the board!");
    }

    // Get the bounding box of the largest contour
    cv::Rect board_rect = cv::boundingRect(contours[largest_contour_index]);

    // Crop the Monopoly board from the image
    cv::Mat cropped_board = current_frame(board_rect);

    // Show results
    cv::imshow("Detected Board", current_frame);
    cv::imshow("Cropped Monopoly Board", cropped_board);
    cv::waitKey(0);

    return cropped_board;
}

void findCameraDetails()
{
    // Chessboard settings
    int chessboard_width = 6;  // Number of inner corners per row
    int chessboard_height = 9; // Number of inner corners per column
    cv::Size board_size(chessboard_width, chessboard_height);

    // Vectors to store object points and image points
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    // Prepare object points (3D coordinates of chessboard corners)
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < chessboard_height; i++) {
        for (int j = 0; j < chessboard_width; j++) {
            objp.push_back(cv::Point3f(j, i, 0)); // Assume z=0 since chessboard is flat
        }
    }

    // Load calibration images
    std::vector<cv::String> images;
    cv::glob("../../../calibration_images/*.jpg", images);  // Ensure images are in this folder

    cv::Mat frame, gray;
    for (const auto& img_file : images) {
        frame = cv::imread(img_file);
        if (frame.empty()) continue;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect chessboard corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, board_size, corners);

        if (found) {
            // Refine corner detection for better accuracy
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

            image_points.push_back(corners);
            object_points.push_back(objp);

            // Draw detected corners
            cv::drawChessboardCorners(frame, board_size, corners, found);
            cv::imshow("Chessboard Detection", frame);
            cv::waitKey(500);
        }
    }

    cv::destroyAllWindows();

    // Camera calibration
    cv::Mat camera_matrix, dist_coeffs, rvecs, tvecs;
    cv::calibrateCamera(object_points, image_points, gray.size(), camera_matrix, dist_coeffs, rvecs, tvecs);

    std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << dist_coeffs << std::endl;

    // Save calibration results
    cv::FileStorage fs("../camera_calibration.yml", cv::FileStorage::WRITE);
    fs << "CameraMatrix" << camera_matrix;
    fs << "DistCoeffs" << dist_coeffs;
    fs.release();

}

cv::Mat undistortImage(string distorted_image_path)
{
    // Load calibration parameters
    cv::FileStorage fs("../camera_calibration.yml", cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeffs;
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistCoeffs"] >> dist_coeffs;
    fs.release();

    // Capture or load a distorted image
    cv::Mat distorted_img = cv::imread(distorted_image_path);
    if (distorted_img.empty()) {
        throw runtime_error("Custom error: Could not load image!");
    }

    // Undistort the image
    cv::Mat undistorted_img;
    cv::undistort(distorted_img, undistorted_img, camera_matrix, dist_coeffs);

    return undistorted_img;
}

int main()
{
    /* maybe try implementing background subtraction so when the game pieces are moved around
    the pieces are shown as the foreground mask, which makes it easier to know the potential locations of the pieces
    then can try doing template matching or SIFT to detect the pieces
    */

    //but first need to calibrate the camera so the board looks like a perfect rectangle
    // findCameraDetails();
    // string distoredImagePath = "../../../main_monopoly_picture.jpg";
    // cv::Mat undistorted_img = undistortImage(distoredImagePath);
    // display_image(undistorted_img, 0.6, "Undistorted Image");
    // cv::destroyWindow("Undistorted Image");



    // string main_monopoly_pic = "../../../main_monopoly_picture.jpg";
    // string scene_image = "../../../SIFT_testing_picture_monopoly.jpg";
    // string angled_main_monopoly_pic = "../../../angled_main_monopoly_picture.jpg";
    // cv::Mat warped_current_video_frame;
    // warped_current_video_frame = testingSIFT(main_monopoly_pic, angled_main_monopoly_pic);
    // cv::Mat cropped_board = crop_out_background(warped_current_video_frame);

    return 0;
}

