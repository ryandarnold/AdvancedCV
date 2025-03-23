#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main()
{
    // diving_video
    cv::Mat resized_frame;
    cv::Mat img = cv::imread("../background.jpg");
    cv::resize(img, resized_frame, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    cv::imshow("test hehe", resized_frame);
    cv::waitKey(0);

    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.