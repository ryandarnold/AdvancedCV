#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "extraFunctions.h"
#include <tuple>
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

void display_video_frame(cv::Mat videoFrameToDisplay, double Scale, string window_name)
{
    //This function just displays a single video frame, and it is up to the caller of this function
    //to determine the delay between frames
    cv::Mat resized_frame;
    cv::resize(videoFrameToDisplay, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
}

cv::Mat SIFT_forGameBoardAlignment(cv::Mat mainBoardTemplateImage, cv::Mat currentFrameImage)
{
    //this function will try to warp the current frame image to match the main board template image
    //using the SIFT algorithm, so that they're aligned as much as possible


    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    sift->detectAndCompute(mainBoardTemplateImage, cv::noArray(), kp1, des1);
    sift->detectAndCompute(currentFrameImage, cv::noArray(), kp2, des2);

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
        throw std::invalid_argument("Error: Homography computation failed!");
    }

    // Warp the second image to align with the original board image
    cv::Mat aligned_scene;
    cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());

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
    // cv::imshow("Detected Board", current_frame);
    // display_image(current_frame, 0.5, "Detected Monopoly Board");
    // cv::imshow("Cropped Monopoly Board", cropped_board);
    // display_image(cropped_board, 0.5, "Cropped Monopoly Board");
    // cv::waitKey(0);

    return cropped_board;
}


void liveVideoOfMonopolyBoard(cv::Mat main_monopoly_image, cv::Mat camera_matrix, cv::Mat dist_coeffs)
{
    //this will be the main loop that will run the game and display the game board
    cv::VideoCapture cap(CAMERA_INDEX);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 60);

    cv::Mat currentFrame, undistorted_current_frame, warped_current_video_frame;
    cv::Mat cropped_board = cv::Mat::zeros(main_monopoly_image.rows, main_monopoly_image.cols, CV_8UC3);  // For a 3-channel image

    double Scale = 0.7;
    int current_frame_count = 0;

    string HAT = "../main_hat_picture_undistorted.jpg";
    cv::Mat gamePiece_HAT =  cv::imread(HAT, cv::IMREAD_COLOR);
    while (true) {
        current_frame_count++;
        cap >> currentFrame; // grab new video frame

        cv::undistort(currentFrame, undistorted_current_frame, camera_matrix, dist_coeffs);
        if (current_frame_count % 3 == 0) //only do SIFT every 3 frames because it is computationally expensive
        {
            warped_current_video_frame = SIFT_forGameBoardAlignment(main_monopoly_image, undistorted_current_frame);
            cropped_board = crop_out_background(warped_current_video_frame);
        }
        display_video_frame(cropped_board, Scale, "Live Camera Feed");

        if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}


void findGamePiece(cv::Mat mainMonopolyBoard, cv::Mat gamePieceTemplate, double threshold = 0.8) {

    // Result matrix
    cv::Mat result;
    cv::matchTemplate(mainMonopolyBoard, gamePieceTemplate, result, cv::TM_CCOEFF_NORMED);

    // Find best match location
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "Best match score: " << maxVal << std::endl;

    // Apply threshold to filter false positives
    if (maxVal >= threshold) {
        // Draw rectangle only if confidence is high
        cv::rectangle(mainMonopolyBoard, maxLoc,
                      cv::Point(maxLoc.x + gamePieceTemplate.cols, maxLoc.y + gamePieceTemplate.rows),
                      cv::Scalar(0, 255, 0), 3);
        std::cout << "Game piece detected at: " << maxLoc << std::endl;
    } else {
        std::cerr << "No good match found! Try lowering the threshold." << std::endl;
    }

    // Show result
    cv::imshow("Detected Game Piece", mainMonopolyBoard);
    cv::waitKey(0);
}



void findHatPieceEdges(cv::Mat mainMonopolyBoard, cv::Mat gamePieceTemplate, double threshold = 0.8)
{
    //NOTE: in this state, this function doesn't work (I think because the threshold for edges
    // apply to both the input template and the game board. I would need to find a 'perfect'
    //template image of the edges hat piece, and then change the threshold on JUST the main board image
    //to get an accurate detection of the hat piece in edge-form

    // Convert both images to grayscale
    cv::Mat board_gray, piece_gray;
    cv::cvtColor(mainMonopolyBoard, board_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gamePieceTemplate, piece_gray, cv::COLOR_BGR2GRAY);

    // Apply Canny edge detection
    cv::Mat board_edges, piece_edges;
    cv::Canny(board_gray, board_edges, 50, 150);  // Adjust thresholds as needed
    cv::Canny(piece_gray, piece_edges, 50, 150);  // Adjust thresholds as needed

    // Perform template matching on edge images
    cv::Mat result;
    cv::matchTemplate(board_edges, piece_edges, result, cv::TM_CCOEFF_NORMED);

    // Find the best match location
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // Apply threshold to filter weak matches
    if (maxVal >= threshold) {
        cv::rectangle(mainMonopolyBoard, maxLoc,
                      cv::Point(maxLoc.x + gamePieceTemplate.cols, maxLoc.y + gamePieceTemplate.rows),
                      cv::Scalar(0, 255, 0), 3);
        std::cout << "Hat piece detected at: " << maxLoc << std::endl;
    } else {
        std::cerr << "No strong match found! Try lowering the threshold." << std::endl;
    }

    // Show results
    cv::imshow("Edge-Detected Board", board_edges);
    cv::imshow("Edge-Detected Hat Piece", piece_edges);
    cv::imshow("Detected Hat Piece", mainMonopolyBoard);
    cv::waitKey(0);
}


cv::Mat equalizeLightingLAB(const cv::Mat& inputImage) {
    // Convert to LAB color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split LAB channels (L = brightness, A & B = color information)
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Apply Otsu’s thresholding on the Luminance (L) channel (without blurring)
    double otsuThreshold = cv::threshold(labChannels[0], labChannels[0], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << "Otsu Threshold for Luminance: " << otsuThreshold << std::endl;

    // Normalize brightness in the L channel
    cv::normalize(labChannels[0], labChannels[0], 0, 255, cv::NORM_MINMAX);

    // Merge channels back
    cv::Mat equalizedLAB;
    cv::merge(labChannels, equalizedLAB);

    // Convert back to BGR color space
    cv::Mat equalizedImage;
    cv::cvtColor(equalizedLAB, equalizedImage, cv::COLOR_Lab2BGR);

    return equalizedImage;
}


cv::Mat adaptiveThresholdLAB(const cv::Mat& inputImage) {
    // Convert to LAB color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split channels (L = brightness, A & B = color)
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Apply adaptive thresholding to the Luminance (L) channel
    // cv::adaptiveThreshold(labChannels[0], labChannels[0], 255,
    //                       cv::ADAPTIVE_THRESH_GAUSSIAN_C,
    //                       cv::THRESH_BINARY, 11, 2);

    cv::adaptiveThreshold(labChannels[0], labChannels[0], 255,
                      cv::ADAPTIVE_THRESH_MEAN_C,  // <== Change from GAUSSIAN_C to MEAN_C
                      cv::THRESH_BINARY, 3, 5);

    // Merge channels back
    cv::Mat processedLAB;
    cv::merge(labChannels, processedLAB);

    // Convert back to BGR
    cv::Mat finalImage;
    cv::cvtColor(processedLAB, finalImage, cv::COLOR_Lab2BGR);

    return finalImage;
}

cv::Mat filterColorHSV(const cv::Mat& inputImage, cv::Scalar lowerBound, cv::Scalar upperBound) {
    // Convert to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Create mask using the color range
    cv::Mat mask;
    cv::inRange(hsvImage, lowerBound, upperBound, mask);

    // Apply the mask to keep only the desired colors
    cv::Mat result;
    cv::bitwise_and(inputImage, inputImage, result, mask);

    return result;
}


cv::Mat filterShinyGrayHSV(const cv::Mat& inputImage) {
    // Convert to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Define HSV range for shiny gray objects
    //HSV; H = 0-180, S = 0-255, V = 0-255
    cv::Scalar lowerGray(0, 0, 50);   // Dark gray
    cv::Scalar upperGray(20, 255, 255); // Slightly shiny gray

    // Create mask
    cv::Mat mask;
    cv::inRange(hsvImage, lowerGray, upperGray, mask);

    // Apply mask to keep only the detected shiny gray objects
    cv::Mat result;
    cv::bitwise_and(inputImage, inputImage, result, mask);

    return result;
}

int main()
{
    //below is for testing----------------------------------------------------------
    // step 1: load in the HAT game piece template image
    // string HAT_path = "../main_hat_picture_undistorted.jpg";
    string HAT_path = "../HATtemplate.jpg";
    cv::Mat HAT_image = cv::imread(HAT_path, cv::IMREAD_COLOR);

    // step 2: load in the current monopoly board that has the HAT game piece on it
    string current_monopoly_board_path = "../singleFrameOfHatOnMonopolyBoard_distorted.jpg";
    cv::Mat current_monopoly_board_image = cv::imread(current_monopoly_board_path, cv::IMREAD_COLOR);

    //step 3: undistort the current monopoly board image
    tuple<cv::Mat, cv::Mat> camera_values = findIntrinsicCameraMatrices();
    cv::Mat camera_matrix = get<0>(camera_values);
    cv::Mat dist_coeffs = get<1>(camera_values);
    cv::Mat undistorted_main_image;
    cv::undistort(current_monopoly_board_image, undistorted_main_image, camera_matrix, dist_coeffs);
    // display_image(undistorted_main_image, 0.5, "Undistorted Monopoly Board");

    //step 4: crop out the background of the undistorted monopoly board image
    cv::Mat cropped_main_monopoly_image = crop_out_background(undistorted_main_image);

    //step 5: now to try to detect the HAT game piece on the undistorted and cropped monopoly board image

    //below does NOT work-------------------------------------------
    // findGamePiece(cropped_main_monopoly_image, HAT_image, 0.15); //uses simple template matching in all color
    // findHatPieceEdges(cropped_main_monopoly_image, HAT_image, 0.01); //uses edge detection for template matching
    //cv::Mat equalizedImage = equalizeLightingLAB(cropped_main_monopoly_image);
    //cv::Mat equalizedImage = adaptiveThresholdLAB(cropped_main_monopoly_image);
    // cv::Mat equalizedImage = filterColorHSV(cropped_main_monopoly_image, lowerBlack, upperBlack);
    //above does NOT work-------------------------------------------



    cv::Mat equalizedImage =  filterShinyGrayHSV(cropped_main_monopoly_image);
    findGamePiece(equalizedImage, HAT_image, 0.3);

    // Display results
    cv::imshow("Original Image", cropped_main_monopoly_image);
    cv::imshow("Lighting Equalized Image", equalizedImage);
    cv::waitKey(0);

    return 0;
    //above is for testing----------------------------------------------------------

    // return 0;
    // 1) note to myself: I still need to test the SIFT at 30FPS and make sure it doesn't lag
    //      and if it does lag, then try only doing SIFT every 5 frames or something -- WORKS DOING SIFT EVERY 3 FRAMES
    //TODO: 2) still need to take pictures of each of the game board pieces and see if SIFT can detect them
    //TODO: 3) then if SIFT can detect the pieces, then need to find the locations (x,y) of each specific piece
    //TODO: 4) then need to make some graphics on screen that will show the user the locations of each piece
    //      relative to the game board so it becomes real-time tracking of game pieces

    /* maybe try implementing background subtraction so when the game pieces are moved around
    the pieces are shown as the foreground mask, which makes it easier to know the potential locations of the pieces
    then can try doing template matching or SIFT to detect the pieces
    */

    //but first need to calibrate the camera so the board looks like a perfect rectangle
    // findCameraDetails();

    //below is main code for the game-------------------------------------------------------
    // string distortedImagePath = "../../../updatedMainMonopolyImage.jpg";
    // tuple<cv::Mat, cv::Mat> camera_values = findIntrinsicCameraMatrices();
    // cv::Mat camera_matrix = get<0>(camera_values);
    // cv::Mat dist_coeffs = get<1>(camera_values);
    //
    // string main_monopoly_pic = "../../../updatedMainMonopolyImage.jpg";
    // string scene_image = "../distorted_angled_main_monopoly_picture.jpg";
    // cv::Mat main_monopoly_image = cv::imread(main_monopoly_pic, cv::IMREAD_COLOR);
    // cv::Mat current_scene_image = cv::imread(scene_image, cv::IMREAD_COLOR);
    //
    // cv::Mat undistorted_main_image;
    // cv::undistort(main_monopoly_image, undistorted_main_image, camera_matrix, dist_coeffs);
    // cv::Mat cropped_main_monopoly_image = crop_out_background(undistorted_main_image);
    // liveVideoOfMonopolyBoard(cropped_main_monopoly_image, camera_matrix, dist_coeffs);
    //above is main code for the game-------------------------------------------------------
    return 0;
}

