#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
/*I have OpenCV version 4.5.5-dev
*/

using namespace std;

int CAMERA_INDEX = 1;

void takeASinglePicture()
{
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
            string filename = "../captured_image_" + std::to_string(imageCount) + ".jpg"; //maybe change to .png
            cv::imwrite(filename, frame); // Save image
            cout << "Image saved as: " << filename << std::endl;
            imageCount++; // Increment image counter
            break;

        }

    }
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}

void takeASingleVideo()
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

void measureFPS()
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

void display_image(cv::Mat original_image, double Scale, string window_name)
{
    cv::Mat resized_frame;
    cv::resize(original_image, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
    cv::waitKey(0);
}

void testingSIFT(string board_template_name, string scene_image_name)
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
        std::cout << "Error: Not enough good matches to compute homography!" << std::endl;
        return;
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
        std::cout << "Error: Homography computation failed!" << std::endl;
        return;
    }

    // Warp the second image to align with the original board image
    cv::Mat aligned_scene;
    cv::warpPerspective(scene_img, aligned_scene, M, board_img.size());

    // Display results
    display_image(board_img, 0.5, "Original Monopoly Board");
    display_image(aligned_scene, 0.5, "aligned Monopoly Board");

    cv::waitKey(0);
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

int main()
{
    // takeASinglePicture();
    // takeASingleVideo();
    // measureFPS();
    // cout << "OpenCV Version: " << CV_VERSION << std::endl;
    // gettingSingleFrameFromAngledVideo();
    string main_monopoly_pic = "../../../main_monopoly_picture.jpg";
    string scene_image = "../../../SIFT_testing_picture_monopoly.jpg";
    string angled_main_monopoly_pic = "../../../angled_main_monopoly_picture.jpg";
    // testingSIFT(main_monopoly_pic, scene_image);

    testingSIFT(main_monopoly_pic, angled_main_monopoly_pic);


    return 0;
}

