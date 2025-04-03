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

void display_video_frame(cv::Mat videoFrameToDisplay, double Scale, string window_name)
{
    //This function just displays a single video frame, and it is up to the caller of this function
    //to determine the delay between frames
    cv::Mat resized_frame;
    cv::resize(videoFrameToDisplay, resized_frame, cv::Size(), Scale, Scale, cv::INTER_AREA);
    cv::imshow(window_name, resized_frame);
}


int main()
{
    cv::VideoCapture cap("../../../VID_20230216_Mug_Rolling_Colors_Glazed_02_Slow_76Seconds.mp4");

    int current_frame_index = 0;
    int start_frame = 53; // This is the first frame to start unrolling the mug
    int end_frame = 231;

    cv::Mat currentFrame;
    cv::Mat unwrappedMug; // accumulate the vertical slices
    while (true) {
        cap >> currentFrame; // grab new video frame
        if (currentFrame.empty()) {
            break; // exit if video is over
        }

        if ((current_frame_index >= start_frame) && (current_frame_index <= end_frame))
        {
            //only process the frames between start_frame and end_frame to unroll the mug

            int slice_offset = 40; // Number of pixels to shift left (center of mug isn't always in the middle of the image)
            int center_x = currentFrame.cols / 2;
            int shifted_x = center_x - slice_offset;
            int slice_width = 10; // sliced image width

            // Calculate the start and end x-coordinates for the vertical slice
            int x_start = max(0, shifted_x - slice_width / 2); //0 in case out of bounds of image
            int x_end = min(currentFrame.cols, x_start + slice_width);

            // Extract vertical strip
            cv::Mat verticalSlice = currentFrame.colRange(x_start, x_end).clone();

            // Crop top and bottom to isolate the mug
            int crop_top = 200; // Adjust this to top of mug
            int crop_height = 700;   // Adjust this to crop
            int y_end = min(crop_top + crop_height, currentFrame.rows);

            cv::Rect mug_rect(0, crop_top, verticalSlice.cols, y_end - crop_top);
            verticalSlice = verticalSlice(mug_rect); //masks the image to the rectangle

            // Concatenate the slice to the unwrapped mug image (horizontally)
            if (unwrappedMug.empty())
            {
                unwrappedMug = verticalSlice.clone();
            }
            else
            {
                // Concatenate the new slice to the existing unwrapped mug image horizontally
                cv::hconcat(unwrappedMug, verticalSlice, unwrappedMug);
            }

            display_video_frame(currentFrame, 0.5, "Mug Video");
            if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
        }
        current_frame_index++;
    }
    
    // Display final result
    if (!unwrappedMug.empty()) {
        display_image(unwrappedMug, 0.5, "Unwrapped Mug Texture");
        cv::imwrite("../unwrapped_mug.png", unwrappedMug); // Saves image
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}
