#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "extraFunctions.h"
#include <tuple>
#include "Player.h" //"" for my own header files, <> for system header files
/*I have OpenCV version 4.5.5-dev
*/

using namespace std;

int CAMERA_INDEX = 0;
int idealColumnAmount = 656; //more than this!
int idealRowAmount = 658; //more than this!

struct LabeledPoint {
    cv::Point point;
    std::string label;
};

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

void drawLabelAbovePoint(cv::Mat& image, const std::string& label, cv::Point point,
                         double fontScale = 0.5, int thickness = 1, cv::Scalar color = cv::Scalar(0, 255, 0))
{
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

    // Position text above the point
    cv::Point textOrg(point.x - textSize.width / 2, point.y - 15);
    cv::putText(image, label, textOrg, fontFace, fontScale, color, thickness);
}

cv::Mat SIFT_forGameBoardAlignment(cv::Mat mainBoardTemplateImage, cv::Mat currentFrameImage)
{
    //this function will try to warp the current frame image to match the main board template image
    //using the SIFT algorithm, so that they're aligned as much as possible


    // Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    sift->detectAndCompute(mainBoardTemplateImage, cv::noArray(), kp1, des1);
    sift->detectAndCompute(currentFrameImage, cv::noArray(), kp2, des2);

    // Use FLANN-based matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);  // Find 2 nearest matches for each descriptor

    // Apply Lowe’s Ratio Test
    std::vector<cv::DMatch> good_matches;
    for (auto& m : knn_matches) {
        if (m[0].distance < 0.75 * m[1].distance) {  // Lowe's Ratio Test
            good_matches.push_back(m[0]);
        }
    }

    // Ensure enough good matches exist for homography
    if (good_matches.size() < 10) {
        //std::cout << "Error: Not enough good matches to compute homography!" << std::endl;
        throw std::invalid_argument("Error: Not enough good matches to compute homography!");
    }

    // Extract keypoint coordinates
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (auto& match : good_matches) {
        src_pts.push_back(kp1[match.queryIdx].pt); // Points in the original Monopoly board image
        dst_pts.push_back(kp2[match.trainIdx].pt); // Corresponding points in the second image
    }

    // Compute homography using RANSAC
    cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC);

    if (M.empty()) {
        throw std::invalid_argument("Error: Homography computation failed!");
    }

    // // Warp the second image to align with the original board image
    cv::Mat aligned_scene;
    cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());

    return aligned_scene; //works but is slow
    // return M;
}


tuple<cv::Mat, cv::Point2f> ORB_forGameBoardAlignment(cv::Mat mainBoardTemplateImage, cv::Mat currentFrameImage)
{
    // 🔹 Step 1: Convert to Grayscale
    cv::Mat edgesTemplate, edgesFrame;
    cv::cvtColor(mainBoardTemplateImage, edgesTemplate, cv::COLOR_BGR2GRAY);
    cv::cvtColor(currentFrameImage, edgesFrame, cv::COLOR_BGR2GRAY);

    // 🔹 Step 2: Apply Canny Edge Detection
    cv::Canny(edgesTemplate, edgesTemplate, 50, 150);
    cv::Canny(edgesFrame, edgesFrame, 50, 150);

    // 🔹 Step 3: Create ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000, 1.2f, 25); // Increase keypoints

    // 🔹 Step 4: Detect keypoints & compute descriptors **on edge images**
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(edgesTemplate, cv::noArray(), kp1, des1);
    orb->detectAndCompute(edgesFrame, cv::noArray(), kp2, des2);

    // 🔹 Step 5: Use Brute-Force Matcher with Cross-Checking
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    // 🔹 Step 6: Sort Matches & Keep the Best
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

    int num_best_matches = std::min(20, (int)matches.size());
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + num_best_matches);

    if (good_matches.size() < 10) {
        throw std::invalid_argument("Error: Not enough good matches to compute homography!");
    }

    // 🔹 Step 7: Extract keypoint coordinates
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (auto& match : good_matches) {
        src_pts.push_back(kp1[match.queryIdx].pt);
        dst_pts.push_back(kp2[match.trainIdx].pt);
    }

    // 🔹 Step 8: Compute Homography
    std::vector<uchar> inliers_mask;
    cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC, 3, inliers_mask);
    if (M.empty()) {
        throw std::invalid_argument("Error: Homography computation failed!");
    }

    // 🔹 Step 9: Warp Image
    cv::Mat aligned_scene;
    cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());

    // 🔹 Step 10: Calculate the Center of Matched Points
    cv::Point2f center(0, 0);
    for (const auto& pt : src_pts) {
        center.x += pt.x;
        center.y += pt.y;
    }
    center.x /= src_pts.size();
    center.y /= src_pts.size();

    return {aligned_scene, center};
}


cv::Mat crop_out_background(cv::Mat current_frame)
{
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(current_frame, gray, cv::COLOR_BGR2GRAY);

    // Apply thresholding to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);

    // Find contours of the image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour (assuming it's the Monopoly board)
    int largest_contour_index = -1;
    double max_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            largest_contour_index = i;
        }
    }

    if (largest_contour_index == -1) {
        // std::cout << "Error: Could not detect the board!" << std::endl;
        throw std::invalid_argument("Error: Could not detect the board!");
    }

    // Get the bounding box of the largest contour
    cv::Rect board_rect = cv::boundingRect(contours[largest_contour_index]);

    // Crop the Monopoly board from the image
    cv::Mat cropped_board = current_frame(board_rect);

    // Show results
    // cv::imshow("Detected Board", current_frame);
    // display_image(current_frame, 0.5, "Detected Monopoly Board");
    // cv::imshow("Cropped Monopoly Board", cropped_board);
    // display_image(cropped_board, 0.5, "Cropped Monopoly Board");
    // cv::waitKey(0);

    return cropped_board;
}


cv::Mat equalizeBoardLighting(const cv::Mat& inputImage)
{
    // Convert from BGR to LAB color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split LAB into individual channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    // Equalize or normalize the L channel (brightness)
    // Option 1: Histogram equalization (stronger, may affect contrast)
    // cv::equalizeHist(labChannels[0], labChannels[0]);

    // Option 2: Normalize (smoother results, recommended for your use)
    cv::normalize(labChannels[0], labChannels[0], 0, 255, cv::NORM_MINMAX);

    // Merge the channels back
    cv::Mat equalizedLab;
    cv::merge(labChannels, equalizedLab);

    // Convert back to BGR
    cv::Mat outputImage;
    cv::cvtColor(equalizedLab, outputImage, cv::COLOR_Lab2BGR);

    return outputImage;
}

cv::Mat equalizeLightingLABColor(const cv::Mat& inputImage) {
    cv::Mat lab;
    cv::cvtColor(inputImage, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);  // L = brightness, a & b = color

    // Blur to estimate lighting pattern on L channel
    cv::Mat floatL, background, diff, normalized;
    labChannels[0].convertTo(floatL, CV_32F);
    cv::GaussianBlur(floatL, background, cv::Size(55, 55), 0);
    diff = floatL - background;
    cv::normalize(diff, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(labChannels[0], CV_8U);  // Replace L with normalized version

    // Merge channels and convert back to BGR
    cv::merge(labChannels, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}



// Helper function to rotate the template image
cv::Mat rotateImage(const cv::Mat& image, int angle)
{
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);

    // Compute bounding box size after rotation
    cv::Rect2f bbox = cv::RotatedRect(center, image.size(), angle).boundingRect2f();

    // Adjust the transformation matrix to fit the whole image
    rotMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // Warp with expanded size
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotMat, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return rotatedImage;
}


cv::Point findPinkPostIt(cv::Mat mainMonopolyBoard, cv::Mat gamePieceTemplate, double threshold = 0.8)
{
    //NOTE: all this function should do is find the PINK post-it note on the game board, and output the center
    //(x,y) points of the PINK post-it note. The calling function should then draw any necessary bounding boxes or
    //red dots to track location

    cv::Mat hsvImage, mask;
    cv::cvtColor(mainMonopolyBoard, hsvImage, cv::COLOR_BGR2HSV);

    // Define HSV range for pink (adjust values if needed)
    //filter out the non-pink colors so template matching works much better
    cv::Scalar lowerPink(10, 50, 140);   // Adjusted lower bound
    cv::Scalar upperPink(30, 180, 255);  // Adjusted upper bound

    cv::inRange(hsvImage, lowerPink, upperPink, mask);
    cv::bitwise_and(mainMonopolyBoard, mainMonopolyBoard, mainMonopolyBoard, mask);

    // 🔹 Resize template to 1/4.5 of original size
    cv::Mat resizedTemplate;
    //1/4.5 works well
    double scaleFactor = 1.0 / 4.5; // had to resize because the template was WAY too big compared to actual game piece
    cv::resize(gamePieceTemplate, resizedTemplate, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    // std::cout << "Resized Template Size: " << resizedTemplate.cols << " x " << resizedTemplate.rows << std::endl;

    double bestMatchScore = 0;
    cv::Point bestMatchLoc;
    cv::Size bestMatchSize;
    int bestRotationAngle = 0;

    // 🔹 Try all rotations (0°, 90°, 180°, 270°)
    std::vector<int> angles = {0, 90, 180, 270};
    // cv::Mat rotatedTemplate = rotateImage(resizedTemplate, 90);
    // display_video_frame(rotatedTemplate, 1, "Rotated Template");
    // display_video_frame(gamePieceTemplate, 1, "Original Template Template");
    for (int angle : angles)
    {
        cv::Mat rotatedTemplate = rotateImage(resizedTemplate, angle);

        // std::cout << "Rotating template by " << angle << " degrees." << std::endl;

        // Perform Template Matching
        cv::Mat result;
        cv::matchTemplate(mainMonopolyBoard, rotatedTemplate, result, cv::TM_CCORR_NORMED);

        // Find best match location
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // std::cout << "Rotation: " << angle << "° | Match score: " << maxVal << std::endl;

        // Check if the current match score is better
        if (maxVal > bestMatchScore)
        {
            bestMatchScore = maxVal;
            bestMatchLoc = maxLoc;
            bestMatchSize = rotatedTemplate.size();
            bestRotationAngle = angle;
        }
    }

    if (bestMatchScore >= threshold)
    {
        // 🔹 Find the Center of the Detected Template
        cv::Point center(bestMatchLoc.x + bestMatchSize.width / 2.0,
                           bestMatchLoc.y + bestMatchSize.height / 2.0);

        // 🔹 Draw the Correctly Rotated Bounding Box
        // cv::RotatedRect rotatedRect(center, bestMatchSize, bestRotationAngle);
        // cv::Point rectPoints[4];
        // rotatedRect.points(rectPoints);  // Get the 4 corner points
        //
        // for (int i = 0; i < 4; i++)
        // {
        //     cv::line(mainMonopolyBoard, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        // }

        return center;
    }
}

cv::Point findAndDisplayPINKPostIt(cv::Mat mainMonopolyBoard, cv::Mat PINK_PostIt_Image, double threshold)
{
    //below is new code for testing without chance card section
    // Clone the board for masking during template matching
    cv::Mat maskedBoard = mainMonopolyBoard.clone();
    // cv::Mat maskedBoard = mainMonopolyBoard;

    // Offsets to shift the mask position
    int offsetX = 125; // move left from right edge
    int offsetY = 120; // move up from bottom edge

    int maskWidth = 125;
    int maskHeight = 125;

    // Compute top-left corner of the mask
    int maskX = mainMonopolyBoard.cols - maskWidth - offsetX;
    int maskY = mainMonopolyBoard.rows - maskHeight - offsetY;

    // Prevent going out of bounds
    maskX = std::max(0, maskX);
    maskY = std::max(0, maskY);

    // Create and apply the mask
    cv::Rect maskRect(maskX, maskY, maskWidth, maskHeight);
    maskedBoard(maskRect) = cv::Scalar(0, 0, 0);  // Apply black mask for template matching

    //NOTE: this rectangle code is just for testing. all it does is draw a black rectangle
    //to determine where the mask is being applied
    // cv::rectangle(mainMonopolyBoard, maskRect, cv::Scalar(0, 0, 0), cv::FILLED);


    // Run template matching on the masked image
    cv::Point pinkPostItCenter = findPinkPostIt(maskedBoard, PINK_PostIt_Image, threshold);

    // If a match is found, draw a red dot
    if (pinkPostItCenter.x != -1 && pinkPostItCenter.y != -1)
    {
        //NOTE: Pink Post it is COLOR RED!!!!
        cv::circle(mainMonopolyBoard, pinkPostItCenter, 10, cv::Scalar(0, 0, 255), -1);
        drawLabelAbovePoint(mainMonopolyBoard, "Player 1", pinkPostItCenter, 0.5, 1, cv::Scalar(0, 0, 255));
    }
    cv::Point intPoint = pinkPostItCenter; //convert cv::Point2f to cv::Point
    //above is new code for testing without chance card section
    static int count = 0;
    if (count == 0)
    {
        std::cout << "Pink Post-it Center: " << pinkPostItCenter << std::endl;
        count++;
    }
    return pinkPostItCenter;

}


cv::Point findBeigePostIt(cv::Mat& mainMonopolyBoard, cv::Mat BEIGE_PostIt_Image, double threshold = 0.8)
{
    // Resize template
    cv::Mat resizedTemplate;
    double scaleFactor = 1.0 / 6;
    cv::resize(BEIGE_PostIt_Image, resizedTemplate, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    double bestMatchScore = 0;
    cv::Point bestMatchLoc;
    cv::Size bestMatchSize;
    int bestRotationAngle = 0;
    cv::Mat bestRotatedTemplate;

    // 🔹 Try all rotations (0°, 90°, 180°, 270°)
    std::vector<int> angles = {0, 90, 180, 270};
    for (int angle : angles)
    {
        // Rotate the template
        cv::Mat rotatedTemplate = rotateImage(resizedTemplate, angle);

        // Perform Template Matching
        cv::Mat result;
        cv::matchTemplate(mainMonopolyBoard, rotatedTemplate, result, cv::TM_CCORR_NORMED);

        // Find best match location
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // Debugging: Print match scores for different angles
        // std::cout << "Angle: " << angle << " | Match score: " << maxVal << std::endl;

        // Check if this rotation is the best match so far
        if (maxVal > bestMatchScore)
        {
            bestMatchScore = maxVal;
            bestMatchLoc = maxLoc;
            bestMatchSize = rotatedTemplate.size();  // Store correct size after rotation
            bestRotationAngle = angle;
            bestRotatedTemplate = rotatedTemplate.clone();
        }
    }

    if (bestMatchScore >= threshold)
    {
        // 🔹 Find the Center of the Best-Matched Template
        cv::Point center(bestMatchLoc.x + bestMatchSize.width / 2.0,
                           bestMatchLoc.y + bestMatchSize.height / 2.0);

        // std::cout << "Best match found at: " << bestMatchLoc
        //           << " | Center: " << center
        //           << " | Rotation: " << bestRotationAngle << ""
        //           << " | Score: " << bestMatchScore << std::endl;

        // 🔹 Draw the Correctly Rotated Bounding Box
        // cv::RotatedRect rotatedRect(center, bestMatchSize, bestRotationAngle);
        // cv::Point2f rectPoints[4];
        // rotatedRect.points(rectPoints);
        //
        // for (int i = 0; i < 4; i++)
        // {
        //     // cout <<"got to point " << rectPoints[i] << endl;
        //     cv::line(mainMonopolyBoard, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
        //     // cv::Rect debugRect(10, 10, 100, 60);  // x=10, y=10, width=100, height=60
        //     // cv::rectangle(mainMonopolyBoard, debugRect, cv::Scalar(57, 150, 255), 4);
        //     // display_video_frame(mainMonopolyBoard, 1, "Debug Rectangle");
        // }
        return center;
    }
}

cv::Mat adjustImageBrightness(const cv::Mat& inputImage, double percentage)
{
    // Convert to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Split into H, S, V channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Calculate brightness adjustment factor
    double factor = 1.0 + (percentage / 100.0);

    // Ensure factor stays positive to avoid inversion
    factor = std::max(0.0, factor);  // if percentage = -100 → factor = 0

    // Apply brightness scaling to V channel
    hsvChannels[2].convertTo(hsvChannels[2], -1, factor, 0);

    // Clip values to [0,255]
    cv::threshold(hsvChannels[2], hsvChannels[2], 255, 255, cv::THRESH_TRUNC);

    // Merge and convert back to BGR
    cv::Mat brightenedHSV, outputImage;
    cv::merge(hsvChannels, brightenedHSV);
    cv::cvtColor(brightenedHSV, outputImage, cv::COLOR_HSV2BGR);

    return outputImage;
}



cv::Point findAndDisplayBEIGEPostIt(cv::Mat mainMonopolyBoard, cv::Mat BEIGE_PostIt_Image, double threshold)
{
    // display_video_frame(BEIGE_PostIt_Image, 1, "Old Beige Post it");
    // cv::Mat newBeigePostItImage = adjustImageBrightness(BEIGE_PostIt_Image, 40);
    // display_video_frame(newBeigePostItImage, 1, "New beige post it ");
    //below is original code but doesn't work with highly reflective surface
    // cv::Mat lightingFixedBoard = equalizeBoardLighting(mainMonopolyBoard);
    cv::Mat normalizedBoardLighting = equalizeLightingLABColor(mainMonopolyBoard);
    // display_video_frame(normalizedBoardLighting, 1, "Lighting Fixed Board");
    // cv::Point2f beigePostItCenter = findBeigePostIt(mainMonopolyBoard, BEIGE_PostIt_Image, threshold);
    cv::Point beigePostItCenter = findBeigePostIt(normalizedBoardLighting , BEIGE_PostIt_Image, threshold);
    cv::Point center(beigePostItCenter.x, beigePostItCenter.y);
    //NOTE: Beige Post it is COLOR MAGENTA!!!!
    cv::circle(mainMonopolyBoard, center, 10, cv::Scalar(255, 0, 255), -1);
    drawLabelAbovePoint(mainMonopolyBoard, "Player 2", beigePostItCenter, 0.5, 1, cv::Scalar(255, 0, 255));
    //above is original code but doesn't work with highly reflective surface
    return beigePostItCenter;

}


vector<cv::Point> findAllGamePieces(cv::Mat current_monopoly_board_image, cv::Mat PINK_PostIt_Image, cv::Mat BEIGE_PostIt_Image,
    Player& player1, Player& player2)
{
    //current_monopoly_board_image is the undistorted and cropped image of the game board (that has the game pieces on it)
    //PINK_PostIt_Image is the game piece that we're trying to detect on the game board (the pink post it)
    vector<cv::Point> playerPositions;
    static int count = 0;

    playerPositions.push_back(findAndDisplayPINKPostIt(current_monopoly_board_image, PINK_PostIt_Image, 0.8));
    playerPositions.push_back(findAndDisplayBEIGEPostIt(current_monopoly_board_image, BEIGE_PostIt_Image, 0.9));

    return playerPositions;
}

void findHatPieceEdges(cv::Mat mainMonopolyBoard, cv::Mat gamePieceTemplate, double threshold = 0.8)
{
    //NOTE: in this state, this function doesn't work (I think because the threshold for edges
    // apply to both the input template and the game board. I would need to find a 'perfect'
    //template image of the edges hat piece, and then change the threshold on JUST the main board image
    //to get an accurate detection of the hat piece in edge-form

    // Convert both images to grayscale
    cv::Mat board_gray, piece_gray;
    cv::cvtColor(mainMonopolyBoard, board_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gamePieceTemplate, piece_gray, cv::COLOR_BGR2GRAY);

    // Apply Canny edge detection
    cv::Mat board_edges, piece_edges;
    cv::Canny(board_gray, board_edges, 50, 150);  // Adjust thresholds as needed
    cv::Canny(piece_gray, piece_edges, 50, 150);  // Adjust thresholds as needed

    // Perform template matching on edge images
    cv::Mat result;
    cv::matchTemplate(board_edges, piece_edges, result, cv::TM_CCOEFF_NORMED);

    // Find the best match location
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // Apply threshold to filter weak matches
    if (maxVal >= threshold) {
        cv::rectangle(mainMonopolyBoard, maxLoc,
                      cv::Point(maxLoc.x + gamePieceTemplate.cols, maxLoc.y + gamePieceTemplate.rows),
                      cv::Scalar(0, 255, 0), 3);
        std::cout << "Hat piece detected at: " << maxLoc << std::endl;
    } else {
        std::cerr << "No strong match found! Try lowering the threshold." << std::endl;
    }

    // Show results
    cv::imshow("Edge-Detected Board", board_edges);
    cv::imshow("Edge-Detected Hat Piece", piece_edges);
    cv::imshow("Detected Hat Piece", mainMonopolyBoard);
    cv::waitKey(0);
}



cv::Mat equalizeLightingLAB(const cv::Mat& inputImage) {
    // Convert to LAB color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split LAB channels (L = brightness, A & B = color information)
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Apply Otsu’s thresholding on the Luminance (L) channel (without blurring)
    double otsuThreshold = cv::threshold(labChannels[0], labChannels[0], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << "Otsu Threshold for Luminance: " << otsuThreshold << std::endl;

    // Normalize brightness in the L channel
    cv::normalize(labChannels[0], labChannels[0], 0, 255, cv::NORM_MINMAX);

    // Merge channels back
    cv::Mat equalizedLAB;
    cv::merge(labChannels, equalizedLAB);

    // Convert back to BGR color space
    cv::Mat equalizedImage;
    cv::cvtColor(equalizedLAB, equalizedImage, cv::COLOR_Lab2BGR);

    return equalizedImage;
}


cv::Mat adaptiveThresholdLAB(const cv::Mat& inputImage) {
    // Convert to LAB color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split channels (L = brightness, A & B = color)
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Apply adaptive thresholding to the Luminance (L) channel
    // cv::adaptiveThreshold(labChannels[0], labChannels[0], 255,
    //                       cv::ADAPTIVE_THRESH_GAUSSIAN_C,
    //                       cv::THRESH_BINARY, 11, 2);

    cv::adaptiveThreshold(labChannels[0], labChannels[0], 255,
                      cv::ADAPTIVE_THRESH_MEAN_C,  // <== Change from GAUSSIAN_C to MEAN_C
                      cv::THRESH_BINARY, 3, 5);

    // Merge channels back
    cv::Mat processedLAB;
    cv::merge(labChannels, processedLAB);

    // Convert back to BGR
    cv::Mat finalImage;
    cv::cvtColor(processedLAB, finalImage, cv::COLOR_Lab2BGR);

    return finalImage;
}

cv::Mat filterColorHSV(const cv::Mat& inputImage, cv::Scalar lowerBound, cv::Scalar upperBound) {
    // Convert to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Create mask using the color range
    cv::Mat mask;
    cv::inRange(hsvImage, lowerBound, upperBound, mask);

    // Apply the mask to keep only the desired colors
    cv::Mat result;
    cv::bitwise_and(inputImage, inputImage, result, mask);

    return result;
}



cv::Mat filterShinyGrayHSV(const cv::Mat& inputImage) {
    // Convert to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Define HSV range for shiny gray objects
    //HSV; H = 0-180, S = 0-255, V = 0-255
    cv::Scalar lowerGray(0, 0, 50);   // Dark gray
    cv::Scalar upperGray(20, 255, 255); // Slightly shiny gray

    // Create mask
    cv::Mat mask;
    cv::inRange(hsvImage, lowerGray, upperGray, mask);

    // Apply mask to keep only the detected shiny gray objects
    cv::Mat result;
    cv::bitwise_and(inputImage, inputImage, result, mask);

    return result;
}

void detectGamePieces(cv::Mat current_monopoly_board_image, cv::Mat PINK_PostIt_Image)
{
    //below does NOT work-------------------------------------------
    // findGamePiece(cropped_main_monopoly_image, HAT_image, 0.15); //uses simple template matching in all color
    // findHatPieceEdges(cropped_main_monopoly_image, HAT_image, 0.01); //uses edge detection for template matching
    //cv::Mat equalizedImage = equalizeLightingLAB(cropped_main_monopoly_image);
    //cv::Mat equalizedImage = adaptiveThresholdLAB(cropped_main_monopoly_image);
    // cv::Mat equalizedImage = filterColorHSV(cropped_main_monopoly_image, lowerBlack, upperBlack);
    //above does NOT work-------------------------------------------




    //*************************************************************************************************************

    // step 2: load in the current monopoly board that has the HAT game piece on it
    // string current_monopoly_board_path = "../singleFrameOfPINK_PostIt_OnMonopolyBoard_LEFT_distorted.jpg";
    // cv::Mat current_monopoly_board_image = cv::imread(current_monopoly_board_path, cv::IMREAD_COLOR);
    //
    // //step 3: undistort the current monopoly board image
    // tuple<cv::Mat, cv::Mat> camera_values = findIntrinsicCameraMatrices();
    // cv::Mat camera_matrix = get<0>(camera_values);
    // cv::Mat dist_coeffs = get<1>(camera_values);
    // cv::Mat undistorted_main_image;
    // cv::undistort(current_monopoly_board_image, undistorted_main_image, camera_matrix, dist_coeffs);
    //
    // //step 4: crop out the background of the undistorted monopoly board image
    // cv::Mat cropped_main_monopoly_image = crop_out_background(undistorted_main_image);
    // display_image(cropped_main_monopoly_image, 0.5, "Cropped Monopoly Board");
    // //step 5: now to try to detect the HAT game piece on the undistorted and cropped monopoly board image
    // findAllGamePieces(cropped_main_monopoly_image, PINK_PostIt_Image);
//*******************************************************************************************************************

    // cv::Mat warped_thing = SIFT_forGameBoardAlignment(cropped_main_monopoly_image, HAT_image);
    //BELOW IS FOR SIFT**************************************************************************************
    // tuple<cv::Mat, cv::Point2f> result=SIFT_forGameBoardAlignment(cropped_main_monopoly_image, HAT_image);
    // // tuple<cv::Mat, cv::Point2f> result = ORB_forGameBoardAlignment(cropped_main_monopoly_image, HAT_image);
    // cv::Mat alignedBoard = std::get<0>(result);
    // cv::Point2f matchCenter = std::get<1>(result);
    // cout << "match center x: " + to_string(matchCenter.x) << endl;
    // cout << "match center y: " + to_string(matchCenter.y) << endl;
    //
    // int rectSize = 20;  // Width and height of the rectangle
    //
    // // Get the top-left and bottom-right corners
    // cv::Point topLeft(matchCenter.x - rectSize / 2, matchCenter.y - rectSize / 2);
    // cv::Point bottomRight(matchCenter.x + rectSize / 2, matchCenter.y + rectSize / 2);
    //
    // // Draw the rectangle on the board image
    // cv::rectangle(cropped_main_monopoly_image, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);  // Green rectangle
    //
    // // Show the image with the rectangle
    // cv::imshow("Detected Center", cropped_main_monopoly_image);
    // cv::waitKey(0);
    //ABOVE IS FOR SIFT**************************************************************************************

}

cv::Point drawBoardCenterCrosshair(cv::Mat& boardImage)
{
    /*This function draws a crosshair at the center of the board image
     *but isn't that useful, since most players sit at the four sides of the board, not at the corners
     **/
    // Step 1: Compute center of the image
    int centerX = boardImage.cols / 2;
    int centerY = boardImage.rows / 2;
    cv::Point center(centerX, centerY);

    // Step 2: Draw black dot at center
    cv::circle(boardImage, center, 5, cv::Scalar(0, 0, 0), -1);  // Black dot

    // Step 3: Vertical line
    cv::line(boardImage,
             cv::Point(centerX, 0),
             cv::Point(centerX, boardImage.rows),
             cv::Scalar(0, 0, 0), 2);

    // Step 4: Horizontal line
    cv::line(boardImage,
             cv::Point(0, centerY),
             cv::Point(boardImage.cols, centerY),
             cv::Scalar(0, 0, 0), 2);

    // Return the center
    return center;
}

cv::Point drawVerticalCenterLine(cv::Mat& boardImage)
{
    // Step 1: Compute horizontal center (X only)
    int centerX = boardImage.cols / 2;
    cv::Point center(centerX, boardImage.rows / 2);

    // Step 2: (Optional) Draw black dot at center
    cv::circle(boardImage, center, 5, cv::Scalar(0, 0, 0), -1);  // Black dot

    // Step 3: Draw vertical line through center
    cv::line(boardImage,
             cv::Point(centerX, 0),                          // top of image
             cv::Point(centerX, boardImage.rows),            // bottom of image
             cv::Scalar(0, 0, 0), 2);                         // black, 2px thick

    return center;
}



cv::Point2f findTenDollarBill(cv::Mat mainMonopolyBoard, cv::Mat TenDollar_Image, double threshold)
{
    // Resize template to 1/6th of its original size
    cv::Mat resizedTemplate;
    double scaleFactor = 1.0 / 10;
    cv::resize(TenDollar_Image, resizedTemplate, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    double bestMatchScore = 0;
    cv::Point bestMatchLoc;
    cv::Size bestMatchSize;
    int bestRotationAngle = 0;
    cv::Mat bestRotatedTemplate;

    // 🔹 Try all rotations (0°, 90°, 180°, 270°)
    std::vector<int> angles = {0, 90, 180, 270};
    for (int angle : angles)
    {
        // Rotate the template
        cv::Mat rotatedTemplate = rotateImage(resizedTemplate, angle);

        // Perform Template Matching
        cv::Mat result;
        cv::matchTemplate(mainMonopolyBoard, rotatedTemplate, result, cv::TM_CCORR_NORMED);

        // Find best match location
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // Debugging: Print match scores for different angles
        // std::cout << "Angle: " << angle << " | Match score: " << maxVal << std::endl;

        // Check if this rotation is the best match so far
        if (maxVal > bestMatchScore)
        {
            bestMatchScore = maxVal;
            bestMatchLoc = maxLoc;
            bestMatchSize = rotatedTemplate.size();  // Store correct size after rotation
            bestRotationAngle = angle;
            bestRotatedTemplate = rotatedTemplate.clone();
        }
    }

    if (bestMatchScore >= threshold)
    {
        // 🔹 Find the Center of the Best-Matched Template
        cv::Point2f center(bestMatchLoc.x + bestMatchSize.width / 2.0,
                           bestMatchLoc.y + bestMatchSize.height / 2.0);

        // std::cout << "Best match found at: " << bestMatchLoc
        //           << " | Center: " << center
        //           << " | Rotation: " << bestRotationAngle << "°"
        //           << " | Score: " << bestMatchScore << std::endl;

        // 🔹 Draw the Correctly Rotated Bounding Box
        cv::RotatedRect rotatedRect(center, bestMatchSize, bestRotationAngle);
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);

        for (int i = 0; i < 4; i++)
            cv::line(mainMonopolyBoard, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

        return center;
    }
}

cv::Point findAndDisplayTenDollarBill(cv::Mat mainMonopolyBoard, cv::Mat TenDollar_Image, double threshold)
{
    cv::Mat normalizedBoardLighting = equalizeLightingLABColor(mainMonopolyBoard);
    // Clone the board for masking during template matching
    cv::Mat maskedBoard = normalizedBoardLighting.clone();

    // Offsets to shift the mask position
    int offsetX = 415; // increase value --> rectangle moves left
    int offsetY = 425; // increase value --> rectangle moves up

    int maskWidth = 125;
    int maskHeight = 125;

    // Compute top-left corner of the mask
    int maskX = mainMonopolyBoard.cols - maskWidth - offsetX;
    int maskY = mainMonopolyBoard.rows - maskHeight - offsetY;

    // Prevent going out of bounds
    maskX = std::max(0, maskX);
    maskY = std::max(0, maskY);

    // Create and apply the mask
    cv::Rect maskRect(maskX, maskY, maskWidth, maskHeight);
    maskedBoard(maskRect) = cv::Scalar(0, 0, 0);  // Apply black mask for template matching

    //NOTE: this rectangle code is just for testing. all it does is draw a black rectangle
    //to determine where the mask is being applied
    // cv::rectangle(mainMonopolyBoard, maskRect, cv::Scalar(0, 0, 0), cv::FILLED);

    // cv::Mat normalizedBoardLighting = equalizeLightingLABColor(mainMonopolyBoard);
    cv::Point2f TenDollarCenter = findTenDollarBill(maskedBoard, TenDollar_Image, threshold);
    cv::Point2f center(TenDollarCenter.x, TenDollarCenter.y);
    //NOTE: TenDollarBill is COLOR GREEN!!!!
    cv::circle(mainMonopolyBoard, center, 10, cv::Scalar(0, 255, 0), -1);
    drawLabelAbovePoint(mainMonopolyBoard, "$10", center, 0.7, 2, cv::Scalar(255, 0, 255));  // magenta
    return center;
}


void determineIfMoneyHasBeenExchanged(cv::Mat mainMonopolyBoard, cv::Mat TenDollar_Image, cv::Point centerOfBoard,
    Player& player1, Player& player2)
{
    static vector<LabeledPoint> trackedPoints;
    static cv::Point previousAverageCenter;
    static int counter = 0;

    //always track the money
    cv::Point tenDollar_Center = findAndDisplayTenDollarBill(mainMonopolyBoard, TenDollar_Image, 0.9);
    //greater than 656 cols (ideal is 658)
    //greater than 658 (ideal is 660)
    //because i found that the best image was 658x660
    if (mainMonopolyBoard.cols > 656 && mainMonopolyBoard.rows > 658) //only track if you have a good image
    {
        //TODO: Ryan, you need to probably find the center point of the image only in here
        // so your algorithm doesn't get confused with the wrong center of the board
        //cout << "good enough image to start tracking" << endl;

        //once the average location of the money, after 5 frames, is to the right of the center,
        //then we know the money has been exchanged
        if (counter < 5)
        {
            //only track up to 5 points at once
            trackedPoints.push_back({tenDollar_Center, "TEN_DOLLARS"});
            counter++;
            // cout << "motherfucker";
        }
        else if (counter % 5 == 0) //update only every 5 frames
        {
            //delete the last point
            // trackedPoints.pop_back();
            trackedPoints.erase(trackedPoints.begin());
            //add the new point
            trackedPoints.push_back({tenDollar_Center, "TEN_DOLLARS"});
            //calculate the average of the last 5 points
            int averageX = 0;
            int averageY = 0;
            for (int i = 0; i < trackedPoints.size(); i++)
            {
                averageX += trackedPoints[i].point.x;
                averageY += trackedPoints[i].point.y;
            }
            averageX = averageX / trackedPoints.size();
            averageY = averageY / trackedPoints.size();

            cv::Point averageCenter(averageX, averageY);
            //TODO: change below 'if statement' if i add more players
            cout << "average center: " << averageCenter << endl;
            cout << "current center: " << tenDollar_Center << endl;
            if (averageCenter.x > centerOfBoard.x && (previousAverageCenter.x < centerOfBoard.x))
            { //money went from left (alice) to right (bob)
                previousAverageCenter = averageCenter; //update previous average center
                player1.deductMoney(10);
                player2.addMoney(10);
                cout << "Alice lost $10, and Bob gained $10!" << endl;
                cout << "player 1 money" << player1.getMoney() << endl;
                cout << "player 2 money" << player2.getMoney() << endl;
            }
            else if (averageCenter.x < centerOfBoard.x && (previousAverageCenter.x > centerOfBoard.x))
            { //money went from right (bob) to left (alice)
                //TODO: do this else if statement (note: this only works with two players so far)
                previousAverageCenter = averageCenter; //update previous average center
                player1.addMoney(10);
                player2.deductMoney(10);
                cout << "Bob lost $10, and Alice gained $10!" << endl;
                cout << "player 1 money" << player1.getMoney() << endl;
                cout << "player 2 money" << player2.getMoney() << endl;
                counter++;
            }
        }


    }

}

void findWhatPropertyPlayersAreOn(cv::Mat mainMonopolyBoard, vector<cv::Point> playerPositions,
    Player& player1, Player& player2)
{
    //now to find what properties each player is on
    cv::Point playerONE_position = playerPositions[0];
    cv::Point playerTWO_position = playerPositions[1];
    // cout << "Player 1 position: " << playerONE_position << endl;
    // cout << "Player 2 position: " << playerTWO_position << endl;

    if (mainMonopolyBoard.cols > idealColumnAmount && mainMonopolyBoard.rows > idealRowAmount)
    {
        //draw a white small vertical line at x = 160, y = 513
        //NOTE: Vermont Avenue x limits: 138 to 191; y limits: (550 to mainMonopolyBoard.rows)

        //TODO: will need to manually check all four corners because they're not consistent
        struct BoardSectionBottom_struct
        {
            string name;
            int leftmost_x;
            int rightmost_x;
            int topmost_y;
        };


        //below values were found from trial and error/testing and are hardcoded for my specific camera

        static const vector<BoardSectionBottom_struct> boardSectionBottom = {
            {"Connecticut Avenue", 86, 137, 550},
            {"Vermont Avenue", 138, 191, 550},
            {"Bottom Chance", 192, 245, 550},
            {"Oriental Avenue", 246, 296, 550},
            {"Reading Railroad", 297, 350, 550},
            {"Income Tax", 351, 405,550},
            {"Baltic Avenue", 406,456,550},
            {"Community Chest", 457, 511, 550},
            {"Mediterranean Avenue", 512, 564,550}
        };

        //below are for the bottom properties
        for (int i = 0; i < boardSectionBottom.size(); i++)
        {
            if ((boardSectionBottom[i].leftmost_x < playerONE_position.x)&&(playerONE_position.x < boardSectionBottom[i].rightmost_x)
                &&(playerONE_position.y > boardSectionBottom[i].topmost_y) )
            {
                if (player1.getCurrentPosition() != boardSectionBottom[i].name)
                {
                    player1.setCurrentPosition(boardSectionBottom[i].name);
                    cout << "Player 1 moved to " << boardSectionBottom[i].name << "!" << endl;
                }
            }
            if ((boardSectionBottom[i].leftmost_x < playerTWO_position.x)&&(playerTWO_position.x < boardSectionBottom[i].rightmost_x)
                &&(playerTWO_position.y > boardSectionBottom[i].topmost_y) )
            {
                if (player2.getCurrentPosition() != boardSectionBottom[i].name)
                {
                    player2.setCurrentPosition(boardSectionBottom[i].name);
                    cout << "Player 2 moved to " << boardSectionBottom[i].name << "!" << endl;
                }
            }
        }

        cv::line(mainMonopolyBoard, cv::Point(564, 550), cv::Point(564, mainMonopolyBoard.rows), cv::Scalar(0, 0, 255), 2);



    }
}

cv::Point genericTemplateMatching(cv::Mat mainMonopolyBoard, cv::Mat templateImage, double scaleFactor,
    vector<int> angles, double threshold, bool drawBoundingBox)
{
    // Resize template
    cv::Mat resizedTemplate;
    // double scaleFactor = 1.0 / 6;
    cv::resize(templateImage, resizedTemplate, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    double bestMatchScore = 0;
    cv::Point bestMatchLoc;
    cv::Size bestMatchSize;
    int bestRotationAngle = 0;
    cv::Mat bestRotatedTemplate;

    // 🔹 Try all rotations (0°, 90°, 180°, 270°)
    // std::vector<int> angles = {0, 90, 180, 270};
    for (int angle : angles)
    {
        // Rotate the template
        cv::Mat rotatedTemplate = rotateImage(resizedTemplate, angle);

        // Perform Template Matching
        cv::Mat result;
        cv::matchTemplate(mainMonopolyBoard, rotatedTemplate, result, cv::TM_CCORR_NORMED);

        // Find best match location
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        // Debugging: Print match scores for different angles
        // std::cout << "Angle: " << angle << " | Match score: " << maxVal << std::endl;

        // Check if this rotation is the best match so far
        if (maxVal > bestMatchScore)
        {
            bestMatchScore = maxVal;
            bestMatchLoc = maxLoc;
            bestMatchSize = rotatedTemplate.size();  // Store correct size after rotation
            bestRotationAngle = angle;
            bestRotatedTemplate = rotatedTemplate.clone();
        }
    }

    if (bestMatchScore >= threshold)
    {
        // 🔹 Find the Center of the Best-Matched Template
        cv::Point center(bestMatchLoc.x + bestMatchSize.width / 2.0,
                           bestMatchLoc.y + bestMatchSize.height / 2.0);

        // std::cout << "Best match found at: " << bestMatchLoc
        //           << " | Center: " << center
        //           << " | Rotation: " << bestRotationAngle << ""
        //           << " | Score: " << bestMatchScore << std::endl;


        if (drawBoundingBox == true)
        {
            // 🔹 Draw the Correctly Rotated Bounding Box
            cv::RotatedRect rotatedRect(center, bestMatchSize, bestRotationAngle);
            cv::Point2f rectPoints[4];
            rotatedRect.points(rectPoints);

            for (int i = 0; i < 4; i++)
            {
                cv::line(mainMonopolyBoard, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }

        }
        return center;
    }
}

int findDiceRoll(cv::Mat mainMonopolyBoard, vector<cv::Mat> allDiceImages)
{
    vector<int> angles = {0, 90, 180, 270};
    vector<cv::Point> diceRollLocations;
    for (int i = 0; i < allDiceImages.size(); i++)
    {
        diceRollLocations.push_back(genericTemplateMatching(mainMonopolyBoard, allDiceImages[i],
            0.2, angles, 0.9, true));
    }
    return 5;
}

void liveVideoOfMonopolyBoard(cv::Mat main_monopoly_image, cv::Mat camera_matrix, cv::Mat dist_coeffs,
    cv::Mat PINK_PostIt_Image, cv::Mat BEIGE_PostIt_Image, cv::Mat TenDollar_Image, vector<cv::Mat> allDiceImages , Player& player1, Player& player2)
{
    //this will be the main loop that will run the game and display the game board
    cv::VideoCapture cap(CAMERA_INDEX);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::Mat currentFrame, undistorted_current_frame, warped_current_video_frame;
    cv::Mat cropped_board = cv::Mat::zeros(main_monopoly_image.rows, main_monopoly_image.cols, CV_8UC3);  // For a 3-channel image

    cv::Mat M = cv::Mat::eye(3, 3, CV_32F);  // 3x3 identity matrix, double precision
    double Scale = 0.7;
    int current_frame_count = 0;

    // string HAT = "../main_hat_picture_undistorted.jpg";
    // cv::Mat gamePiece_HAT =  cv::imread(HAT, cv::IMREAD_COLOR);
    while (true) {
        current_frame_count++;
        cap >> currentFrame; // grab new video frame
        // cv::Mat M;
        //undistorts from wide-angle to normal/flat camera
        cv::undistort(currentFrame, undistorted_current_frame, camera_matrix, dist_coeffs);
        // if (current_frame_count % 30 == 0) //try SIFT only once a second
        // {
        //     M = SIFT_forGameBoardAlignment(main_monopoly_image, undistorted_current_frame);
        // }
        if (current_frame_count % 3 == 0) //only do SIFT every 3 frames because it is computationally expensive
        {
            // Warp the second image to align with the original board image
            // cv::Mat warped_current_video_frame;
            // cv::warpPerspective(undistorted_current_frame, warped_current_video_frame, M, main_monopoly_image.size());

            warped_current_video_frame = SIFT_forGameBoardAlignment(main_monopoly_image, undistorted_current_frame);
            cropped_board = crop_out_background(warped_current_video_frame);
            //need to rotate because SIFT changes the rotation
            cv::rotate(cropped_board, cropped_board, cv::ROTATE_90_COUNTERCLOCKWISE);
            // Display image size
            std::string sizeText = "Size: " + std::to_string(cropped_board.cols) + "x" + std::to_string(cropped_board.rows);
            cv::putText(cropped_board, sizeText, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                        1.0, cv::Scalar(0, 255, 255), 1);

            vector<cv::Point> player_positions = findAllGamePieces(cropped_board, PINK_PostIt_Image, BEIGE_PostIt_Image, player1, player2);

            cv::Point centerOfboard = drawVerticalCenterLine(cropped_board);
            determineIfMoneyHasBeenExchanged(cropped_board, TenDollar_Image, centerOfboard,
                player1, player2);
            findWhatPropertyPlayersAreOn(cropped_board, player_positions, player1, player2);
            int current_dice_roll_number = findDiceRoll(cropped_board, allDiceImages);


        }
        display_video_frame(cropped_board, Scale, "Live Camera Feed");

        if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}

int main()
{
    //below is for testing----------------------------------------------------------

    // takeASinglePicture(CAMERA_INDEX, "../singleBEIGE_PostIt_uncropped.jpg");
    // takeASinglePicture(CAMERA_INDEX, "../singleFrameOfRED_PostIt_OnMonopolyBoard_LEFT_distorted.jpg");

    // takeASinglePicture(CAMERA_INDEX, "../DICE_value_ONE_uncropped.jpg");
    // return 0;
    //above is for testing----------------------------------------------------------

    // 1) note to myself: I still need to test the SIFT at 30FPS and make sure it doesn't lag
    //      and if it does lag, then try only doing SIFT every 5 frames or something -- WORKS DOING SIFT EVERY 3 FRAMES
    //TODO: 2) still need to take pictures of each of the game board pieces and see if SIFT can detect them
    //TODO: 3) then if SIFT can detect the pieces, then need to find the locations (x,y) of each specific piece
    //TODO: 4) then need to make some graphics on screen that will show the user the locations of each piece
    //      relative to the game board so it becomes real-time tracking of game pieces

    /* maybe try implementing background subtraction so when the game pieces are moved around
    the pieces are shown as the foreground mask, which makes it easier to know the potential locations of the pieces
    then can try doing template matching or SIFT to detect the pieces
    */

    //but first need to calibrate the camera so the board looks like a perfect rectangle
    // findCameraDetails();

    //below is main code for the game-------------------------------------------------------

    //Step 1: load in the camera intrinsics
    tuple<cv::Mat, cv::Mat> camera_values = findIntrinsicCameraMatrices();
    cv::Mat camera_matrix = get<0>(camera_values);
    cv::Mat dist_coeffs = get<1>(camera_values);

    //Step 2: load in main monopoly board template to use for SIFT (later)
    string main_monopoly_pic = "../updatedMainMonopolyImage.jpg";
    // string scene_image = "../distorted_angled_main_monopoly_picture.jpg";
    cv::Mat main_monopoly_image = cv::imread(main_monopoly_pic, cv::IMREAD_COLOR);
    // cv::Mat current_scene_image = cv::imread(scene_image, cv::IMREAD_COLOR);

    //Step 3: undistort the main monopoly board image template
    cv::Mat undistorted_main_image;
    cv::undistort(main_monopoly_image, undistorted_main_image, camera_matrix, dist_coeffs);

    //Step 4: crop out the background of the undistorted monopoly board image so the whole image is just monopoly board
    cv::Mat cropped_main_monopoly_image = crop_out_background(undistorted_main_image);

    //Step 5: load in the game piece templates
    //load in PINK post it piece
    string PINK_PostIt_Path = "../singlePINK_PostIt_cropped.jpg";
    cv::Mat PINK_PostIt_Image = cv::imread(PINK_PostIt_Path, cv::IMREAD_COLOR);

    //load in BEIGE post it piece
    string BEIGE_PostIt_Path = "../singleBEIGE_PostIt_cropped.jpg";
    cv::Mat BEIGE_PostIt_Image = cv::imread(BEIGE_PostIt_Path, cv::IMREAD_COLOR);

    //Step 6: load in the money templates
    //load in 10 dollar bill
    string TenDollarBill_Path = "../singleTenDollarBill_cropped.jpg";
    cv::Mat TenDollar_Image = cv::imread(TenDollarBill_Path, cv::IMREAD_COLOR);

    Player player1("Alice", 10, "GO");  // name = "Alice", starting money = 10
    Player player2("Bob", 0, "GO");    // name = "Bob", starting money = 0

    cout <<"Player 1: " << player1.getName() << ", starting Money: " << player1.getMoney() << ", position: " << player1.getCurrentPosition() << endl;
    cout <<"Player 2: " << player2.getName() << ", startingMoney: " << player2.getMoney() << ", position: " << player2.getCurrentPosition() << endl;

    cv::Mat dice_ONE = cv::imread("../DICE_value_ONE_cropped.jpg", cv::IMREAD_COLOR);
    vector<cv::Mat> allDiceImages;
    allDiceImages.push_back(dice_ONE);


    liveVideoOfMonopolyBoard(cropped_main_monopoly_image, camera_matrix, dist_coeffs,
        PINK_PostIt_Image, BEIGE_PostIt_Image, TenDollar_Image, allDiceImages, player1, player2);

    //above is main code for the game-------------------------------------------------------
    return 0;
}

