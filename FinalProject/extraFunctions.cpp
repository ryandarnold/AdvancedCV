#include "extraFunctions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <tuple>
#include "Player.h"
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
    std::vector<std::vector<cv::Point3f>> object_points; // 3D points in real world space
    std::vector<std::vector<cv::Point2f>> image_points;  //same 3D points but in 2D image space

    // Prepare object points (3D coordinates of chessboard corners)
    // aka Creates a 2D grid of 3D points on the Z=0 plane.
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < chessboard_height; i++) {
        for (int j = 0; j < chessboard_width; j++) {
            objp.push_back(cv::Point3f(j, i, 0)); // Assume z=0 since chessboard is flat, and z is always the same
        }
    }

    // Load calibration images
    std::vector<cv::String> images;
    cv::glob("../../../calibration_images_new_camera/*.jpg", images);  // Ensure images are in this folder

    cv::Mat frame, gray;
    for (const auto& img_file : images)
    {
        frame = cv::imread(img_file); //grab current calibration images (which are chessboard images)
        if (frame.empty()) continue;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //chessboard calibration works best in grayscale

        // Detect chessboard corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, board_size, corners); //main function that detects the chessboard corners

        if (found) {
            //now 'corners' matrix will have the 2D coordinates of the chessboard corners

            //build up many object --> image mappings
            image_points.push_back(corners); //put all corner coordinates into the image_points vector
            object_points.push_back(objp);   //

            // Draw detected corners
            cv::drawChessboardCorners(frame, board_size, corners, found);
            cv::imshow("Chessboard Detection", frame);
            cv::waitKey(500);
        }
    }

    cv::destroyAllWindows();

    // Camera calibration
    //camera_matrix is a 3x3 matrix that contains the intrinsic parameters of the camera
    //ex:
    // [[fx, 0, cx],
    //  [0, fy, cy],
    //  [0, 0, 1]]
    //where fx and fy are the focal lengths in pixels, and cx and cy are the coordinates of the optical center (principal point).

    //dist_coeffs is a vector of distortion coefficients that describe the lens distortion = [k1, k2, p1, p2, k3]
    //where k1, k2, k3 are radial distortion coefficients, and p1, p2 are tangential distortion coefficients
    //(radial distortion means straight lines in real world appear to be curved in the image)
    //(and tangential distortion means the lens and image plane are not perfectly parallel. i.e. the image isn't flat)

    //rvecs tells how the camera is rotated with respect to the chessboard in each image
    //tvecs tells where the chessboard is in 3D space relative to the camera
    cv::Mat camera_matrix, dist_coeffs, rvecs, tvecs;
    cv::calibrateCamera(object_points, image_points, gray.size(), camera_matrix, dist_coeffs, rvecs, tvecs);

    std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << dist_coeffs << std::endl;

    // Save calibration results
    cv::FileStorage fs("../NEW_CAMERA_camera_calibration.yml", cv::FileStorage::WRITE);
    fs << "CameraMatrix" << camera_matrix;
    fs << "DistCoeffs" << dist_coeffs;
    fs.release();
}

tuple<cv::Mat, cv::Mat> findIntrinsicCameraMatrices()
{
    //This function loads the camera matrix and distortion coefficients from a file

    // Load calibration parameters
    cv::FileStorage fs("../NEW_CAMERA_camera_calibration.yml", cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeffs;
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistCoeffs"] >> dist_coeffs;
    fs.release();

    return {camera_matrix, dist_coeffs};
}

void testVideoWithUndistortingEachFrame(int CAMERA_INDEX, cv::Mat camera_matrix, cv::Mat dist_coeffs)
{
    //This function is just for testing to see if undistorting each frame with the homography found by
    // SIFT works
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
    //This function allows you to take multiple pictures with the camera and save them to the specified path
    // in one go
    cout << "starting to take multiple pictures" << endl;
    for (int i = 0; i < numImages; i++)
    {
        takeASinglePicture(CAMERA_INDEX, to_string(i) + imageName);
        cout << "finished taking picture: " << i << endl;
    }
}

void takeASinglePicture(int CAMERA_INDEX, string imagePathANDnameANDextension)
{
    //This function just takes a single picture with the camera and saves it to the specified path
    //NOTE: this function does NOT undistort the image using the camera intrinsics to account for barrel warping
    cout << "camera index: " + CAMERA_INDEX << endl;
    cv::VideoCapture cap(CAMERA_INDEX); // Open the default camera (0 for the first camera)
    std::cout << "Current Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Current Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Current FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    //NOTE: BELOW IS FOR OLD CAMERA
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080); //best resolution but way too laggy for real-time detection
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    // cap.set(cv::CAP_PROP_FPS, 60);
    //NOTE: ABOVE IS FOR OLD CAMERA


    cap.set(cv::CAP_PROP_FRAME_WIDTH, 3840); //best
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 2160);
    cap.set(cv::CAP_PROP_FPS, 30);


    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1000); //test
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 2160);
    // cap.set(cv::CAP_PROP_FPS, 30);


    int imageCount = 1; // Counter for saved images
    cv::Mat frame;
    double Scale = 0.3;
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
    //This function tests taking a video at different frames per second
    cv::VideoCapture cap(CAMERA_INDEX); // Open the default camera (0 for the first camera)
    cv::VideoWriter writer;

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
    //This function measures the actual frames per second (FPS) of the camera feed
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

static std::vector<std::string> loadNames(const std::string& p){
    std::vector<std::string> n; std::ifstream f(p);
    for (std::string s; std::getline(f, s);) if(!s.empty()) n.push_back(s);
    return n;
}

int simpleYOLOv5Detection()
{

    const std::string model = "assets/yolov5s_455.onnx";
    const std::string namesPath = "assets/coco-labels-2014_2017.txt";
    const std::string imgPath   = "assets/test_thingy.jpg"; // any test image

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) { std::cerr << "No image\n"; return 1; }

    // Preprocess (simple resize to 640x640; good enough for a smoke test)
    cv::Mat blob = cv::dnn::blobFromImage(img, 1/255.0, {640,640}, cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());   // outs[0]: 1x25200x85

    const float confTh = 0.25f, nmsTh = 0.45f;
    float xFac = img.cols / 640.0f, yFac = img.rows / 640.0f;

    std::vector<int>    classIds, indices;
    std::vector<float>  confidences;
    std::vector<cv::Rect> boxes;

    const float* data = (float*)outs[0].data;
    for (int i = 0; i < 25200; ++i) {
        float obj = data[4];
        if (obj < confTh) { data += 85; continue; }

        // best class
        int   bestId = 0; float best = 0.f;
        for (int c = 5; c < 85; ++c) if (data[c] > best){ best = data[c]; bestId = c - 5; }
        float conf = obj * best;
        if (conf >= confTh) {
            float cx = data[0], cy = data[1], w = data[2], h = data[3];
            int left = int((cx - 0.5f*w) * xFac);
            int top  = int((cy - 0.5f*h) * yFac);
            int ww   = int(w * xFac);
            int hh   = int(h * yFac);
            boxes.emplace_back(left, top, ww, hh);
            confidences.push_back(conf);
            classIds.push_back(bestId);
        }
        data += 85;
    }

    cv::dnn::NMSBoxes(boxes, confidences, confTh, nmsTh, indices);
    auto names = loadNames(namesPath);

    for (int idx : indices) {
        cv::rectangle(img, boxes[idx], {0,255,0}, 2);
        std::string label = (classIds[idx] < (int)names.size() ? names[classIds[idx]] : "obj")
                            + cv::format(" %.2f", confidences[idx]);
        cv::putText(img, label, boxes[idx].tl() + cv::Point(0,-3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 2);
    }

    cv::imshow("YOLO test", img);
    cv::waitKey();
}








