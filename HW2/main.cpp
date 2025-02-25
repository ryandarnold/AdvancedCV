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
            cv::imshow("Background Model", background);
            display_video_frame(background, 0.5, "Background Model");
        }


        //
        // cv::imshow("example RUN hehe", resized_frame);
        if (cv::waitKey(33) >= 0) break; //hardcoded 33ms delay between frames
    }
    // Save the background image
    if (!background.empty()) {
        cv::imwrite("../background.jpg", background);
        std::cout << "Background image saved as background.jpg" << std::endl;
    }
}

int main()
{
    part2();

    return 0;
}