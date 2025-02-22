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

void testingSIFT()
{
    cv::Mat board_img = cv::imread("main_monopoly_picture.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat scene_img = cv::imread("SIFT_testing_picture_monopoly.jpg", cv::IMREAD_GRAYSCALE);

    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2; //list of keypoints in each image
    cv::Mat des1, des2;
    sift->detectAndCompute(board_img, cv::noArray(), kp1, des1);
    sift->detectAndCompute(scene_img, cv::noArray(), kp2, des2);

    // Match features using FLANN
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    // Keep only the best matches
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + 50);

    if (good_matches.size() > 10) {  // Ensure enough matches
        std::vector<cv::Point2f> src_pts, dst_pts;
        for (auto& match : good_matches) {
            src_pts.push_back(kp1[match.queryIdx].pt);
            dst_pts.push_back(kp2[match.trainIdx].pt);
        }

        // Compute homography
        cv::Mat M = cv::findHomography(src_pts, dst_pts, cv::RANSAC);

        if (!M.empty()) {
            // Warp the scene to align the board correctly
            cv::Mat aligned_board;
            cv::warpPerspective(scene_img, aligned_board, M, board_img.size());

            cv::imshow("Warped Monopoly Board", aligned_board);
            cv::waitKey(0);
        }
    }

}

int main()
{
    //takeASinglePicture();
    // takeASingleVideo();
    // measureFPS();
    // cout << "OpenCV Version: " << CV_VERSION << std::endl;
    testingSIFT();
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.