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
}

void part2()
{
    //step 1: open the video
    cv::VideoCapture cap;
    cap.open("../../../Walking_through_Back_Yard.mp4");
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Create background subtractor (MOG2)
    //creates a smart pointer that allocates and deallocates memory for you (no need to call delete)
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();

    cv::Mat fgMask, background;

    cv::Mat frame;
    while (true)
    {   //runs the video in a loop
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction
        bgSubtractor->apply(frame, fgMask); //white = foreground, black = background

        bgSubtractor->getBackgroundImage(background); // Get the background model

        display_video_frame(frame, 0.5, "Original Frame");
        display_video_frame(fgMask, 0.5, "Foreground Mask");

        if (!background.empty()) {
            //display background frame/image as it updates
            display_video_frame(background, 0.5, "Background Model");
        }

        if (cv::waitKey(33) >= 0) break; //hardcoded 33ms delay between frames
    }

    if (!background.empty()) {
        // Save the background image
        cv::imwrite("../background.jpg", background);
        std::cout << "Background image saved as background.jpg" << std::endl;
    }
}


void part3()
{
    // Open the input video
    cv::VideoCapture cap("../../../Walking_through_Back_Yard.mp4");

    // Create Background Subtractor smart pointer
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2(300,
    15,true);

    // Get frame info to save the video
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Define the codec and create VideoWriter object
    cv::VideoWriter outputVideo("../../../DrKinsman_noBackground.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, fgMask, background, processedFrame;

    while (true) {
        cap >> frame; // Read frame
        if (frame.empty()) break; // Break at end of video

        // Apply main background subtraction
        bgSubtractor->apply(frame, fgMask);

        // elliptical kernel creation
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

        // Close holes in the foreground (helps fill missing parts of objects)
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);

        // Create a new background Green is [0,255,0]
        //size is the size of the original input frame, type is the type of the frame, and Scalar is the color
        cv::Mat newBackground(frame.size(), frame.type(), cv::Scalar(0, 255, 0)); // Green background

        // Convert mask to 3 channels because the mask is a single gray channel :(
        // cv::Mat fgMask3Ch;
        // cv::cvtColor(fgMask, fgMask3Ch, cv::COLOR_GRAY2BGR);

        // uses 'fgMask' as a mask on the current 'frame' and stores the result in 'processedFrame'
        frame.copyTo(processedFrame, fgMask); // Keep only foreground pixels

        // Replace background where mask is black
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                if (fgMask.at<uchar>(i, j) == 0) { //if the current mask is a background pixel (0 for black)
                    //cv::Vec3B tells OpenCV to treat the pixel as a 3 channel pixel instead of gray/single channel
                    //replace the pixel with the new background pixel
                    processedFrame.at<cv::Vec3b>(i, j) = newBackground.at<cv::Vec3b>(i, j);
                }
            }
        }

        // Write the frame to the output video
        outputVideo.write(processedFrame);

        // Display
        display_video_frame(frame, 0.5, "Original Frame");
        display_video_frame(fgMask, 0.5, "Foreground Mask");
        cv::imshow("Processed Video", processedFrame);
        display_video_frame(processedFrame, 0.5, "Processed Video");

        if (cv::waitKey(25) >= 0) break; //33 for 30fps, 25 for faster playback (but doesn't really work)
    }

    cap.release();
    outputVideo.release();
    cv::destroyAllWindows();

    std::cout << "Output video saved as DrKinsman_noBackground.avi" << std::endl;
}

int main()
{
    //part2();
    //part3();

    return 0;
}