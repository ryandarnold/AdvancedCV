#include "extraFunctions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;



void takeMultiplePictures(string cameraCalibration_path, int CAMERA_INDEX, string imageName, int numImages)
{
    // string cameraCalibration_path = "../../../calibration_images/"; //all .jpg images in folder
    // string imageName = "imageHEHE";
    // takeMultiplePictures(cameraCalibration_path, CAMERA_INDEX, imageName, 10);
    cout << "starting to take multiple pictures" << endl;
    for (int i = 0; i < numImages; i++)
    {
        takeASinglePicture(cameraCalibration_path, CAMERA_INDEX, imageName + to_string(i));
        cout << "finished taking picture: " << i << endl;
    }

}


void takeASinglePicture(string cameraCalibration_path, int CAMERA_INDEX, string imageName)
{
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
            string filename = cameraCalibration_path + imageName + ".jpg";
            cv::imwrite(filename, frame); // Save image
            cout << "Image saved as: " << filename << std::endl;
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


