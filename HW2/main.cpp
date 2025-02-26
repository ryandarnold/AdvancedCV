#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void display_image(cv::Mat original_image, double Scale, string window_name)
{
    cv::Mat resized_frame;
    cv::resize(original_image, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
    cv::waitKey(0);
}

void display_video_frame(cv::Mat original_image, double Scale, string window_name)
{
    cv::Mat resized_frame;
    cv::resize(original_image, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
    // cv::waitKey(0);
}

void part2()
{
    //
    //step 1: open the video
    cv::VideoCapture cap;
    cap.open("../../../Walking_through_Back_Yard.mp4");
    double fps = cap.get(cv::CAP_PROP_FPS);
    // std::cout << "Frame rate (FPS): " << fps << std::endl;
    int delay_between_frames = round(1000.0 / fps);
    // cout << "ms delay between frames: " << delay_between_frames << endl;

    // Create background subtractor (MOG2)
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

    cv::Mat fgMask, background;

    cv::Mat frame;
    double Scale = 0.6;
    while (true)
    {   //runs the video in a loop
        cap >> frame;
        if (frame.empty()) break;

        bgSubtractor->apply(frame, fgMask); // Apply background subtraction

        bgSubtractor->getBackgroundImage(background); // Get the background model


        display_video_frame(frame, 0.5, "Original Frame");
        display_video_frame(fgMask, 0.5, "Foreground Mask");

        if (!background.empty()) {
            // cv::imshow("Background Model", background);
            display_video_frame(background, 0.5, "Background Model");
        }

        if (cv::waitKey(33) >= 0) break; //hardcoded 33ms delay between frames
    }
    // Save the background image
    if (!background.empty()) {
        cv::imwrite("../background.jpg", background);
        std::cout << "Background image saved as background.jpg" << std::endl;
    }
}


void part3()
{
    // Open the input video
    cv::VideoCapture cap("../../../Walking_through_Back_Yard.mp4");

    // Create Background Subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

    // Get frame properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Define the codec and create VideoWriter object
    cv::VideoWriter outputVideo("../../../DrKinsman_noBackground.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, fgMask, background, processedFrame;

    while (true) {
        cap >> frame; // Read frame
        if (frame.empty()) break; // Break at end of video

        // Apply background subtraction
        bgSubtractor->apply(frame, fgMask);

        // Create a new background (Choose color: Green [0,255,0], Blue [255,0,0], or Gray [128,128,128])
        cv::Mat newBackground(frame.size(), frame.type(), cv::Scalar(0, 255, 0)); // Green background

        // Convert mask to 3 channels to match frame format
        cv::Mat fgMask3Ch;
        cv::cvtColor(fgMask, fgMask3Ch, cv::COLOR_GRAY2BGR);

        // Extract foreground using the mask
        frame.copyTo(processedFrame, fgMask); // Keep only foreground pixels

        // Replace background where mask is black
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                if (fgMask.at<uchar>(i, j) == 0) { // Background detected
                    processedFrame.at<cv::Vec3b>(i, j) = newBackground.at<cv::Vec3b>(i, j);
                }
            }
        }

        // Write the frame to the output video
        outputVideo.write(processedFrame);

        // Display
        //cv::imshow("Original Frame", frame);
        display_video_frame(frame, 0.5, "Original Frame");
        //cv::imshow("Foreground Mask", fgMask);
        display_video_frame(fgMask, 0.5, "Foreground Mask");
        cv::imshow("Processed Video", processedFrame);
        display_video_frame(processedFrame, 0.5, "Processed Video");


        // Press 'q' to exit early
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    outputVideo.release();
    cv::destroyAllWindows();

    std::cout << "Output video saved as DrKinsman_noBackground.avi" << std::endl;

}

int main()
{
    // part2();
    part3();

    return 0;
}