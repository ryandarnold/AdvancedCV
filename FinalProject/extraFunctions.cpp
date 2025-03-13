#include "extraFunctions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;

void findCameraDetails()
{   //NOTE: this function only needs to be run once per camera
    /* This function calibrates the camera using a chessboard pattern.
     * The camera matrix and distortion coefficients
     * are saved to a file for later use.
    */

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

tuple<cv::Mat, cv::Mat> findIntrinsicCameraMatrices()
{
    //TODO: only call this function once, and then output the camera_matrix and dist_coeffs to the calling function
    // because accessing file storage will be really slow if you do it at 30fps

    // Load calibration parameters
    cv::FileStorage fs("../camera_calibration.yml", cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeffs;
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistCoeffs"] >> dist_coeffs;
    fs.release();

    return {camera_matrix, dist_coeffs};
}

void testVideoWithUndistortingEachFrame(int CAMERA_INDEX, cv::Mat camera_matrix, cv::Mat dist_coeffs)
{
    cv::VideoCapture cap(CAMERA_INDEX); // Open the default camera (0 for the first camera)

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 30);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Camera FPS: " << fps << std::endl;


    bool recording = false;
    cv::Mat frame;
    cv::Mat undistorted_frame;
    while (true) {
        cap >> frame; // new frame into frame matrix
        cv::undistort(frame, undistorted_frame, camera_matrix, dist_coeffs);
        cv::imshow("Live Camera Feed", undistorted_frame);


        // Check if any key is pressed
        int key = cv::waitKey(1);
        if (key >= 0) // Any key detected
        {
            break;
        }

    }
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}

void takeMultiplePictures(int CAMERA_INDEX, string imageName, int numImages)
{
    // string cameraCalibration_path = "../../../calibration_images/"; //all .jpg images in folder
    // string imageName = "imageHEHE";
    // takeMultiplePictures(cameraCalibration_path, CAMERA_INDEX, imageName, 10);
    cout << "starting to take multiple pictures" << endl;
    for (int i = 0; i < numImages; i++)
    {
        takeASinglePicture(CAMERA_INDEX, to_string(i) + imageName);
        cout << "finished taking picture: " << i << endl;
    }
}

void takeASinglePicture(int CAMERA_INDEX, string imagePathANDnameANDextension)
{
    //NOTE: this function does NOT undistort the image using the camera intrinsics to account for barrel warping
    cout << "camera index: " + CAMERA_INDEX << endl;
    cv::VideoCapture cap(CAMERA_INDEX); // Open the default camera (0 for the first camera)
    std::cout << "Current Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Current Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Current FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080); //best resolution but way too laggy for real-time detection
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 60);


    int imageCount = 1; // Counter for saved images
    cv::Mat frame;
    double Scale = 0.5;
    while (true) {
        cap >> frame; // new frame into frame matrix

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
        cv::imshow("Camera Feed", resized_frame); // Show the frame
        // Wait for a key press (1 ms delay), and capture image if any key is pressed
        int key = cv::waitKey(1);
        if (key >= 0) {  // Any key pressed
            // string filename = "../captured_image_" + std::to_string(imageCount) + ".jpg"; //maybe change to .png
            // string filename = imagePathANDnameANDextension;
            cv::imwrite(imagePathANDnameANDextension, frame); // Save image
            cout << "Image saved as: " << imagePathANDnameANDextension << std::endl;
            imageCount++; // Increment image counter
            break;
        }

    }
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}

void takeASingleVideo(int CAMERA_INDEX)
{
    cv::VideoCapture cap(CAMERA_INDEX); // Open the default camera (0 for the first camera)
    cv::VideoWriter writer;
    // std::cout << "Current Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    // std::cout << "Current Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    // std::cout << "Current FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;


    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080); //best resolution but way too laggy for real-time detection
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 30);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Camera FPS: " << fps << std::endl;


    bool recording = false;
    // int imageCount = 1; // Counter for saved images
    cv::Mat frame;
    double Scale = 0.5;

    while (true) {
        cap >> frame; // new frame into frame matrix
        cv::imshow("Live Camera Feed", frame);

        // If recording, write frame to video
        if (recording) {
            writer.write(frame);
        }

        // Check if any key is pressed
        int key = cv::waitKey(1);
        if (key >= 0) { // Any key detected
            if (recording == false) {
                std::string filename = "../recorded_video.avi";
                writer.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
                if (!writer.isOpened()) {
                    std::cout << "Error: Could not open video file for writing!" << std::endl;
                    return;
                }
                std::cout << "Recording started..." << std::endl;
                recording = true;
            } else {
                writer.release();
                std::cout << "Recording stopped. Video saved as 'recorded_video.avi'." << std::endl;
                recording = false;
                // stopVideo = true;
                break;
            }

        }

    }
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}

void gettingSingleFrameFromAngledVideo()
{
    /*all this function does is grab a single frame from the baseline monopoly board that is angled for more
     *realistic testing of the SIFT algorithm
     */
    // cv::VideoCapture cap(CAMERA_INDEX);
    cv::VideoCapture cap;
    cap.open("../../../angled_baseline_monopoly.avi");
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Frame rate (FPS): " << fps << std::endl;
    int delay_between_frames = round(1000.0 / fps);
    cout << "ms delay between frames: " << delay_between_frames << endl;

    cv::Mat frame;
    int frame_to_capture = 1;
    int curr_frame_number = 1;
    for (;;)
    {   //runs the video in a loop
        cap >> frame;
        if (frame.empty()) break;
        if (curr_frame_number == frame_to_capture)
        {
            string filename = "../../../captured_image.jpg";
            cv::imwrite(filename, frame); // Save image
            cout << "Image saved as: " << filename << std::endl;
            curr_frame_number++;
        }
        cv::imshow("example RUN hehe", frame);
        if (cv::waitKey(delay_between_frames) >= 0) break;
    }
}

void measureFPS(int CAMERA_INDEX)
{
    cv::VideoCapture cap(CAMERA_INDEX); // Open default camera
    cv::Mat frame;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        cap >> frame; // Capture a new frame
        if (frame.empty()) break;

        cv::imshow("Camera Feed", frame);
        frame_count++;

        // Calculate FPS every 30 frames
        if (frame_count == 30) {
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
            double fps = frame_count / duration;
            std::cout << "Actual FPS: " << fps << std::endl;

            // Reset timer and frame count
            start = std::chrono::high_resolution_clock::now();
            frame_count = 0;
        }

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}

//below is working SIFT code that i'm storing in case the new chatgpt code doesn't work!
// cv::Mat SIFT_forGameBoardAlignment(cv::Mat mainBoardTemplateImage, cv::Mat currentFrameImage)
// {
//     //this function will try to warp the current frame image to match the main board template image
//     //using the SIFT algorithm, so that they're aligned as much as possible
//
//
//     // Create SIFT detector
//     cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//
//     // Detect keypoints and compute descriptors
//     std::vector<cv::KeyPoint> kp1, kp2;
//     cv::Mat des1, des2;
//     sift->detectAndCompute(mainBoardTemplateImage, cv::noArray(), kp1, des1);
//     sift->detectAndCompute(currentFrameImage, cv::noArray(), kp2, des2);
//
//     // Use FLANN-based matcher
//     cv::FlannBasedMatcher matcher;
//     std::vector<std::vector<cv::DMatch>> knn_matches;
//     matcher.knnMatch(des1, des2, knn_matches, 2);  // Find 2 nearest matches for each descriptor
//
//     // Apply Lowe’s Ratio Test
//     std::vector<cv::DMatch> good_matches;
//     for (auto& m : knn_matches) {
//         if (m[0].distance < 0.75 * m[1].distance) {  // Lowe's Ratio Test
//             good_matches.push_back(m[0]);
//         }
//     }
//
//     // Ensure enough good matches exist for homography
//     if (good_matches.size() < 10) {
//         //std::cout << "Error: Not enough good matches to compute homography!" << std::endl;
//         throw std::invalid_argument("Error: Not enough good matches to compute homography!");
//     }
//
//     // Extract keypoint coordinates
//     std::vector<cv::Point2f> src_pts, dst_pts;
//     for (auto& match : good_matches) {
//         src_pts.push_back(kp1[match.queryIdx].pt); // Points in the original Monopoly board image
//         dst_pts.push_back(kp2[match.trainIdx].pt); // Corresponding points in the second image
//     }
//
//     // Compute homography using RANSAC
//     cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC);
//
//     if (M.empty()) {
//         throw std::invalid_argument("Error: Homography computation failed!");
//     }
//
//     // Warp the second image to align with the original board image
//     cv::Mat aligned_scene;
//     cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());
//
//     return aligned_scene;
// }
//above is working SIFT code that i'm storing in case the new chatgpt code doesn't work!

//below is SIFT code that outputs the warped image and the center point of the matched points
// tuple<cv::Mat, cv::Point2f> SIFT_forGameBoardAlignment(cv::Mat mainBoardTemplateImage, cv::Mat currentFrameImage)
// {
//     // 🔹 Step 1: Convert to Grayscale
//     cv::Mat edgesTemplate, edgesFrame;
//     cv::cvtColor(mainBoardTemplateImage, edgesTemplate, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(currentFrameImage, edgesFrame, cv::COLOR_BGR2GRAY);
//
//     // 🔹 Step 2: Apply Canny Edge Detection
//     cv::Canny(edgesTemplate, edgesTemplate, 50, 150);
//     cv::Canny(edgesFrame, edgesFrame, 50, 150);
//
//     // 🔹 Step 3: Create SIFT detector
//     cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//
//     // 🔹 Step 4: Detect keypoints & compute descriptors **on edge-detected images**
//     std::vector<cv::KeyPoint> kp1, kp2;
//     cv::Mat des1, des2;
//     sift->detectAndCompute(edgesTemplate, cv::noArray(), kp1, des1);
//     sift->detectAndCompute(edgesFrame, cv::noArray(), kp2, des2);
//
//     // 🔹 Step 5: Use FLANN-based matcher
//     cv::FlannBasedMatcher matcher;
//     std::vector<std::vector<cv::DMatch>> knn_matches;
//     matcher.knnMatch(des1, des2, knn_matches, 2);
//
//     // 🔹 Step 6: Apply Lowe’s Ratio Test
//     std::vector<cv::DMatch> good_matches;
//     for (auto& m : knn_matches) {
//         if (m[0].distance < 0.5 * m[1].distance) {
//             good_matches.push_back(m[0]);
//         }
//     }
//
//     // 🔹 Step 7: Ensure enough good matches exist for homography
//     if (good_matches.size() < 10) {
//         throw std::invalid_argument("Error: Not enough good matches to compute homography!");
//     }
//
//     // 🔹 Step 8: Extract keypoint coordinates
//     std::vector<cv::Point2f> src_pts, dst_pts;
//     for (auto& match : good_matches) {
//         src_pts.push_back(kp1[match.queryIdx].pt); // Points in mainBoardTemplateImage
//         dst_pts.push_back(kp2[match.trainIdx].pt); // Points in currentFrameImage
//     }
//
//     // 🔹 Step 9: Compute homography using RANSAC
//     cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC);
//     if (M.empty()) {
//         throw std::invalid_argument("Error: Homography computation failed!");
//     }
//
//     // 🔹 Step 10: Warp the current frame to align with the template
//     cv::Mat aligned_scene;
//     cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());
//
//     // 🔹 Step 11: Calculate the center of matched points in `mainBoardTemplateImage`
//     cv::Point2f center(0, 0);
//     for (const auto& pt : src_pts) {
//         center.x += pt.x;
//         center.y += pt.y;
//     }
//     center.x /= src_pts.size();
//     center.y /= src_pts.size();
//
//     return {aligned_scene, center};
// }

//above is SIFT code that outputs the warped image and the center point of the matched points