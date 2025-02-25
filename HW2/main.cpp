#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void part2()
{
    //
    //step 1: open the video
    cv::VideoCapture cap;
    cap.open("../../../Walking_through_Back_Yard.mp4");
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Frame rate (FPS): " << fps << std::endl;
    int delay_between_frames = round(1000.0 / fps);
    cout << "ms delay between frames: " << delay_between_frames << endl;

    cv::Mat frame;
    double Scale = 0.6;
    while (true)
    {   //runs the video in a loop
        cap >> frame;
        if (frame.empty()) break;
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
        cv::imshow("example RUN hehe", resized_frame);
        if (cv::waitKey(delay_between_frames) >= 0) break;
    }
}

int main()
{
    part2();

    return 0;
}