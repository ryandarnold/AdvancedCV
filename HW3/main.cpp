/*
 * This program reads a video file and creates a mosaic of the video frames with the background model
 * The program uses a background subtractor to find the diver in the video frame
 * The program then creates a mosaic of the video frames with the background model
 * The program also keeps track of the parts of the video that have already been occupied by divers
 * so each diver doesn't 'touch' each other in the mosaic
 *
 * ***NOTE***: I used the "diving_video_far__board_477E38120CAF" video (the 2nd video)
 */


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

cv::Mat getBrightestImage(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3)
{
    /*This function takes in 3 images and returns the brightest image
     */
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
    /*
     * This program reads a video file and creates a mosaic of the video frames with the background model
    */

    int number_of_brightest_frames = 4; //finding 1 brightest frame out of 4 frames

    cv::VideoCapture cap("../../../diving_video.mp4");
    int totalFrameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)); //finds the total number of frames in the video

    //ignore the last 'x' number of frames if they're not a multiple of 'number_of_brightest_frames'
    int totalFramesToProcess = totalFrameCount - (totalFrameCount % number_of_brightest_frames);
    cout << "Total frames: " << totalFrameCount << endl;

    //create a background subtractor object
    cv::Ptr<cv::BackgroundSubtractor> background_subtractor = cv::createBackgroundSubtractorMOG2();

    //load the background model .jpg image
    cv::Mat backgroundModel = cv::imread("../background_Diving_model.jpg");

    cv::Mat currentFrame; //current frame from the video
    cv::Mat videoMosaic = backgroundModel.clone(); //initialize the video mosaic with the background model

    //need to track the parts of the video that have already been occupied by divers
    // so each diver doesn't 'touch' each other in the mosaic
    cv::Mat occupiedArea = cv::Mat::zeros(backgroundModel.size(), CV_8UC1);;

    std::vector<cv::Mat> buffer; //buffer to store the last 'number_of_brightest_frames' frames
    cv::Mat brightestFrame;
    int frameCount = 1; //frame count to know when to process the brightest frame

    while (true) {
        cap >> currentFrame; // grab new video frame
        if (currentFrame.empty()) {
            break; // exit if video is over
        }

        buffer.push_back(currentFrame); //add the current frame to the buffer to
        //keep track of brightest number_of_brightest_frames

        if ((frameCount % number_of_brightest_frames == 0) && (frameCount < totalFramesToProcess))
        {
            //frameCount % number_of_brightest_frames == 0 means we have gone 'number_of_brightest_frames' frames
            //frameCount < totalFramesToProcess means we only go up to a multiple
            //of number_of_brightest_frames (i was getting getting errors if i don't include this)

            //now to check the brightest frame in the last "number_of_brightest_frames"
            cv::Mat brightestFrameInBuffer = getBrightestImage(buffer[0], buffer[1], buffer[2]);

            cv::Mat fgMask; //everything white is the foreground mask
            background_subtractor->apply(brightestFrameInBuffer, fgMask); //apply background subtraction

            //from here down, the fgMask is the foreground (white), and everything else is black

            // Clean the mask to get better foreground object of the diver
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::erode(fgMask, fgMask, kernel, cv::Point(-1, -1), 1);
            cv::dilate(fgMask, fgMask, kernel, cv::Point(-1, -1), 6);

            //threshold because the mask is not binary, because of the erode and dilate kernel
            //i.e. parts of the mask become gray because of the erode and dilate
            cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);

            // mask out water because it messes with below 'diver area' and 'overlap area' calculations
            int waterlineY = 1000; // adjust for different video heights
            cv::Rect waterLine(0, waterlineY, fgMask.cols, fgMask.rows - waterlineY);
            fgMask(waterLine) = 0; // set water area to black always

            cv::Mat overlap; //finds the overlap between the occupied area and the fgMask
            cv::bitwise_and(occupiedArea, fgMask, overlap);

            //needs to find the area of the diver and the overlap area
            int diverArea = cv::countNonZero(fgMask);
            int overlapArea = cv::countNonZero(overlap);

            //only add the diver to the mosaic if the diver area is greater than 10000
            //and the overlap area is 0 (i.e. the divers aren't touching in mid-air)
            if (diverArea > 1000 && overlapArea == 0) { //anything less than 14000ish works to get all divers
                //copies brightestFrameInBuffer, with the fgMask as the mask, to the videoMosaic
                brightestFrameInBuffer.copyTo(videoMosaic, fgMask);

                // adds the current 'fgMask' to the 'occupiedArea' so the next diver doesn't touch the current diver
                cv::bitwise_or(occupiedArea, fgMask, occupiedArea);
            }

            display_video_frame(brightestFrameInBuffer, 0.5, "Diving Video");
            display_video_frame(fgMask, 0.5, "Current Foreground Mask");

            if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
            buffer.clear();
        }

        frameCount++;
    }

    cv::destroyAllWindows();
    display_video_frame(videoMosaic, 0.5, "Final Diver Mosaic");
    // cv::imwrite("../Final Diver Mosaic.jpg", videoMosaic); // Save to disk
    cv::waitKey(0);
    return 0;
}

