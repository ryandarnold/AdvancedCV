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

    cv::VideoCapture cap("../../../OpticalFlowVideo1.mp4");

    cv::Mat prev_frame, prev_gray;
    cap >> prev_frame;
    if (prev_frame.empty()) return -1;

    // Convert to grayscale because LK and goodFeatures are easier to compute with grayscale
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> prev_pts, next_pts; // Points to track for goodFeaturesToTrack
        std::vector<uchar> status; // Status of each point: 1 if the point was found, 0 otherwise
        std::vector<float> err; // Error for each point (not important for now)

        // Detect corners in previous frame to input into Lucas-Kanade method.
        // i.e. goodFeatureToTrack finds the corners, and LK tracks them (to track their movement change,
        //you must know where they are in the first place)
        cv::goodFeaturesToTrack(prev_gray, prev_pts, 30, 0.1, 15);

        // optical flow (Lucas-Kanade)
        cv::Size winSize(11, 11); //window to look for changes around the point (11x11 pixels)
        int maxLevel = 3; // Number of pyramid levels (for larger motion estimation)
        cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, next_pts, status, err,
            winSize, maxLevel);

        for (size_t i = 0; i < prev_pts.size(); ++i)
        {
            //loop through all previous point
            if (status[i]) //check if the point was found at the previous_points location
            {
                cv::Point2f p1 = prev_pts[i];
                cv::Point2f p2 = next_pts[i];

                cv::Point2f flow = p2 - p1;
                float mag = cv::norm(flow); //finds the magnitude of the flow vector

                if (mag < 0.5 || mag > 20) continue; //skips the point if the flow is too small or too large

                cv::arrowedLine(frame, p1, p1 + 5.0f * flow, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.25);
            }
        }

        display_video_frame(frame, 0.5, "Lucas-Kanade Optical Flow");

        int key = cv::waitKey(30);
        if (key == 27) break;  // ESC to quit
        prev_gray = gray.clone();
    }
    return 0;
}