#include <iostream>
#include <opencv2/opencv.hpp>

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main()
{
    cv::Mat image = cv::imread("../camel_hehe.jpg"); // Replace with your image path

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

    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.