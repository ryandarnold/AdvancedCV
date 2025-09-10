#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

#include <tuple>
#include "Player.h" //"" for my own header files, <> for system header files
#include "extraFunctions.h"
/*I have OpenCV version 4.5.5-dev
*/

using namespace std;

int CAMERA_INDEX = 1;
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

    // Detects all keypoints in main board template image and current frame image
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    sift->detectAndCompute(mainBoardTemplateImage, cv::noArray(), kp1, des1);
    sift->detectAndCompute(currentFrameImage, cv::noArray(), kp2, des2);

    // Use FLANN-based matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);  // Find 2 nearest matches for each descriptor

    // Apply Lowe’s Ratio Test to find only the BEST matches
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
    std::vector<cv::Point2f> src_pts, dst_pts; //src is main template, dst is current frame
    for (auto& match : good_matches) {
        src_pts.push_back(kp1[match.queryIdx].pt); // Points in the original Monopoly board image
        dst_pts.push_back(kp2[match.trainIdx].pt); // Corresponding points in the second image
    }

    // Compute homography using RANSAC after getting all source and destination points
    cv::Mat M = cv::findHomography(dst_pts, src_pts, cv::RANSAC);

    if (M.empty())
    {
        throw std::invalid_argument("Error: Homography computation failed!");
    }

    return M; //return JUST the homography matrix so i can later do the warping in the main function

    // // Warp the second image to align with the original board image
    //cv::Mat aligned_scene;
    //cv::warpPerspective(currentFrameImage, aligned_scene, M, mainBoardTemplateImage.size());

    //return aligned_scene; //works but is slow

}



cv::Mat crop_out_background(cv::Mat current_frame)
{
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(current_frame, gray, cv::COLOR_BGR2GRAY);

    // Apply thresholding to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);

    // Find contours of the image using binary image
    std::vector<std::vector<cv::Point>> contours; //vector of vector of points that are contours
    //stores relationship between contours
    //i.e. next contour, previous contour, child contour, parent contour
    // in a hierarchy
    std::vector<cv::Vec4i> hierarchy;
    //RETR_EXTERNAL means we only want the outermost contours (main board outline)
    //CHAIN_APPROX_SIMPLE uses compression to save memory
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    // Find the largest contour
    int largest_contour_index = -1; //stores the index of the largest contour
    double max_area = 0; // stores the area of the largest contour
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]); //finds area of current contour
        if (area > max_area)
        {
            max_area = area; //only used to find the largest contour
            largest_contour_index = i;
        }
    }

    if (largest_contour_index == -1) {
        // std::cout << "Error: Could not detect the board!" << std::endl;
        throw std::invalid_argument("Error: Could not detect the board!");
    }

    // Get the bounding box of the largest contour
    //contours[largest_contour_index] is a vector of 2D points outlining that contour
    //boundingRect returns the smallest rectangle that contains all the points in the contour
    //so if the contour is generally a square, the bounding box will be a square
    cv::Rect board_rect = cv::boundingRect(contours[largest_contour_index]);

    // Crop the Monopoly board from the image
    cv::Mat cropped_board = current_frame(board_rect);

    return cropped_board;
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

    double bestMatchScore = 0;
    cv::Point bestMatchLoc;
    cv::Size bestMatchSize;
    int bestRotationAngle = 0;

    // 🔹 Try all rotations (0°, 90°, 180°, 270°)
    std::vector<int> angles = {0, 90, 180, 270};
    for (int angle : angles)
    {
        cv::Mat rotatedTemplate = rotateImage(resizedTemplate, angle);

        // Perform Template Matching
        cv::Mat result;
        cv::matchTemplate(mainMonopolyBoard, rotatedTemplate, result, cv::TM_CCORR_NORMED);

        // Find min and max values in "result", and their locations
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);


        // Check if the current match score is better
        if (maxVal > bestMatchScore)
        {
            bestMatchScore = maxVal; // Update best match score
            bestMatchLoc = maxLoc; // Update best match location. NOTE: bestMatchLoc is the top-left corner of the bounding box
            bestMatchSize = rotatedTemplate.size(); // Store correct size after rotation
            bestRotationAngle = angle;
        }
    }

    if (bestMatchScore >= threshold)
    {
        // 🔹 Find the Center of the Detected Template
        cv::Point center(bestMatchLoc.x + bestMatchSize.width / 2.0,
                           bestMatchLoc.y + bestMatchSize.height / 2.0);

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

        // 🔹 Draw the Correctly Rotated Bounding Box
        cv::RotatedRect rotatedRect(center, bestMatchSize, bestRotationAngle);
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        return center;
    }
}


cv::Point findAndDisplayBEIGEPostIt(cv::Mat mainMonopolyBoard, cv::Mat BEIGE_PostIt_Image, double threshold)
{

    cv::Point beigePostItCenter = findBeigePostIt(mainMonopolyBoard , BEIGE_PostIt_Image, threshold);
    cv::Point center(beigePostItCenter.x, beigePostItCenter.y);
    //NOTE: Beige Post it is COLOR MAGENTA!!!!
    cv::circle(mainMonopolyBoard, center, 10, cv::Scalar(255, 0, 255), -1);
    drawLabelAbovePoint(mainMonopolyBoard, "Player 2", beigePostItCenter, 0.5, 1, cv::Scalar(255, 0, 255));
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
    // Resize template
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
        //once the average location of the money, after 5 frames, is to the right of the center,
        //then we know the money has been exchanged
        if (counter < 5)
        {
            //only track up to 5 points at once
            trackedPoints.push_back({tenDollar_Center, "TEN_DOLLARS"});
        }
        else if (counter % 5 == 0) //update only every 5 frames
        {
            //delete the last point
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

            if (averageCenter.x > centerOfBoard.x && (previousAverageCenter.x < centerOfBoard.x))
            { //money went from left (alice) to right (bob)
                // previousAverageCenter = averageCenter; //update previous average center
                player1.deductMoney(10);
                player2.addMoney(10);
                cout << "Alice lost $10, and Bob gained $10!" << endl;
                cout << "player 1 money" << player1.getMoney() << endl;
                cout << "player 2 money" << player2.getMoney() << endl;
            }
            else if (averageCenter.x < centerOfBoard.x && (previousAverageCenter.x > centerOfBoard.x))
            {
                //money went from right (bob) to left (alice)
                player1.addMoney(10);
                player2.deductMoney(10);
                cout << "Bob lost $10, and Alice gained $10!" << endl;
                cout << "player 1 money" << player1.getMoney() << endl;
                cout << "player 2 money" << player2.getMoney() << endl;

            }
            previousAverageCenter = averageCenter; //update previous average center
        }
        counter++;
    }
}

void findWhatPropertyPlayersAreOn(cv::Mat mainMonopolyBoard, vector<cv::Point> playerPositions,
    Player& player1, Player& player2)
{
    //now to find what properties each player is on
    cv::Point playerONE_position = playerPositions[0];
    cv::Point playerTWO_position = playerPositions[1];

    if (mainMonopolyBoard.cols > idealColumnAmount && mainMonopolyBoard.rows > idealRowAmount)
    {
        //draw a white small vertical line at x = 160, y = 513
        //NOTE: Vermont Avenue x limits: 138 to 191; y limits: (550 to mainMonopolyBoard.rows)
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
        // cv::line(mainMonopolyBoard, cv::Point(564, 550), cv::Point(564, mainMonopolyBoard.rows), cv::Scalar(0, 0, 255), 2);

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





void liveVideoOfMonopolyBoard(cv::Mat main_monopoly_image, cv::Mat camera_matrix, cv::Mat dist_coeffs,
    cv::Mat PINK_PostIt_Image, cv::Mat BEIGE_PostIt_Image, cv::Mat TenDollar_Image, Player& player1, Player& player2)
{


    //this will be the main loop that will run the game and display the game board
    cv::VideoCapture cap(CAMERA_INDEX);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); // worked very well
    cap.set(cv::CAP_PROP_FPS, 60);

    cv::Mat currentFrame, undistorted_current_frame, warped_current_video_frame;
    cv::Mat cropped_board = cv::Mat::zeros(main_monopoly_image.rows, main_monopoly_image.cols, CV_8UC3);  // For a 3-channel image

    //cv::Mat M = cv::Mat::eye(3, 3, CV_32F);  // 3x3 identity matrix, double precision
    double Scale = 0.7;
    int current_frame_count = 0;
    cv::Mat M;
    while (true) {
        cap >> currentFrame; // grab new video frame
        cv::undistort(currentFrame, undistorted_current_frame, camera_matrix, dist_coeffs);

        if (current_frame_count % 60 == 0) //only do SIFT every 3 frames because it is computationally expensive
        {
            M = SIFT_forGameBoardAlignment(main_monopoly_image, undistorted_current_frame);
            //cv::Mat warped_current_video_frame;
            //cv::warpPerspective(undistorted_current_frame, warped_current_video_frame, M, main_monopoly_image.size());
        }

        cv::warpPerspective(undistorted_current_frame, warped_current_video_frame, M, main_monopoly_image.size());
        // Warp the second image to align with the original board image
        //warped_current_video_frame = SIFT_forGameBoardAlignment(main_monopoly_image, undistorted_current_frame);
        cropped_board = crop_out_background(warped_current_video_frame);
        //need to rotate because SIFT changes the rotation
        cv::rotate(cropped_board, cropped_board, cv::ROTATE_90_COUNTERCLOCKWISE);
        // Display image size
        std::string sizeText = "Size: " + std::to_string(cropped_board.cols) + "x" + std::to_string(cropped_board.rows);
        cv::putText(cropped_board, sizeText, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 255), 1);
        //
        //vector<cv::Point> player_positions = findAllGamePieces(cropped_board, PINK_PostIt_Image, BEIGE_PostIt_Image, player1, player2);

        //cv::Point centerOfboard = drawVerticalCenterLine(cropped_board);
        //determineIfMoneyHasBeenExchanged(cropped_board, TenDollar_Image, centerOfboard,
        //player1, player2);
        //findWhatPropertyPlayersAreOn(cropped_board, player_positions, player1, player2);



        display_video_frame(cropped_board, Scale, "Live Camera Feed");
        //display_video_frame(undistorted_current_frame, Scale, "Live Camera Feed");


        current_frame_count++;
        if (int key = cv::waitKey(33); key >= 0) { break;} // displays at 30FPS
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Close OpenCV windows
}


int main()
{

    //Step 1: load in the camera intrinsics
    tuple<cv::Mat, cv::Mat> camera_values = findIntrinsicCameraMatrices();
    cv::Mat camera_matrix = get<0>(camera_values);
    cv::Mat dist_coeffs = get<1>(camera_values);

    //Step 2: load in main monopoly board template to use for SIFT (later)
    string main_monopoly_pic = "../updatedMainMonopolyImage.jpg";
    cv::Mat main_monopoly_image = cv::imread(main_monopoly_pic, cv::IMREAD_COLOR);

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



    liveVideoOfMonopolyBoard(cropped_main_monopoly_image, camera_matrix, dist_coeffs,
        PINK_PostIt_Image, BEIGE_PostIt_Image, TenDollar_Image, player1, player2);

    //above is main code for the game-------------------------------------------------------
    return 0;
}

