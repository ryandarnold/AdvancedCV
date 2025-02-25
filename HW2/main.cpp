#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void partA()
{
    /*Display any image using OpenCV.
    Even a small one will do.
    */
    cv::Mat image = cv::imread("../../../camel_hehe.jpg"); // Replace with your image path

    // Check if the image was loaded properly
    if (image.empty()) {
        std::cerr << "Error: Unable to load image." << std::endl;
    }

    // Resize image to 50% of its original size
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    // Display the image
    cv::imshow("Image", resized_image);

    // Wait for a key press and close the window
    cv::waitKey(0);
}

int main()
{
    cout << "hehe";
    partA();

    // string main_monopoly_pic = "../../../main_monopoly_picture.jpg";
    // string scene_image = "../../../SIFT_testing_picture_monopoly.jpg";
    // string angled_main_monopoly_pic = "../../../angled_main_monopoly_picture.jpg";
    // cv::Mat warped_current_video_frame;
    // warped_current_video_frame = testingSIFT(main_monopoly_pic, angled_main_monopoly_pic);
    // cv::Mat cropped_board = crop_out_background(warped_current_video_frame);

    return 0;
}