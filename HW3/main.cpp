#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void display_video_frame(cv::Mat videoFrameToDisplay, double Scale, string window_name)
{
    //This function just displays a single video frame, and it is up to the caller of this function
    //to determine the delay between frames
    cv::Mat resized_frame;
    cv::resize(videoFrameToDisplay, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
}

cv::Mat getBrightestImage(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3) {
    cv::Mat gray1, gray2, gray3;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img3, gray3, cv::COLOR_BGR2GRAY);

    double mean1 = cv::mean(gray1)[0];
    double mean2 = cv::mean(gray2)[0];
    double mean3 = cv::mean(gray3)[0];

    if ((mean1 >= mean2) && (mean1 >= mean3))
    {
        return img1;
    }
    else if ((mean2 >= mean1) && (mean2 >= mean3))
    {
        return img2;
    }
    else
    {
        return img3;
    }
}

int main()
{
    int number_of_brightest_frames = 4; //4, 6 works well except for beginning;

    cv::VideoCapture cap("../../../diving_video.mp4");
    double fps = cap.get(cv::CAP_PROP_FPS);
    // double framecount = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int totalFrameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    //ignore the last 'x' number of frames if they're not a multiple of 'number_of_brightest_frames'
    int totalFramesToProcess = totalFrameCount - (totalFrameCount % number_of_brightest_frames);
    cout << "Total frames: " << totalFrameCount << endl;
    cout << "Frame rate (FPS): " << fps << endl;


    cv::Mat currentFrame;
    std::vector<cv::Mat> buffer;
    cv::Mat brightestFrame;
    int frameCount = 1;
    while (true) {
        cap >> currentFrame; // grab new video frame
        buffer.push_back(currentFrame); //add the current frame to the buffer to keep track of brightest three frames

        if ((frameCount % number_of_brightest_frames == 0) && (frameCount < totalFramesToProcess))
        {
            //now to check the brightest frame in the last three frames
            cv::Mat brightestFrameInBuffer = getBrightestImage(buffer[0], buffer[1], buffer[2]);
            //
            display_video_frame(brightestFrameInBuffer, 0.5, "Diving Video");
            if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
            buffer.clear();
        }

        frameCount++;
        // display_video_frame(currentFrame, 0.5, "Diving Video"); //displays normal video
        // if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
    }

    return 0;
}

