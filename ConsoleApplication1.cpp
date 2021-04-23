#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main() {

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("C:\\Users\\preet\\Desktop\\CodeChallenge\\Opencv_tutorial\\v1.mp4");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while (1) {

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        //medianBlur(gray, gray, 5);
        vector<Vec3f> circles;
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
            gray.rows / 16,  // change this value to detect circles with different distances to each other
            100, 30, 1, 30 // change the last two parameters
       // (min_radius & max_radius) to detect larger circles
        );
        for (size_t i = 0; i < circles.size(); i++)
        {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            // circle center
            circle(gray, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
            // circle outline
            int radius = c[2];
            circle(gray, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
        }
        imshow("detected circles", gray);
        // Display the resulting frame
        //imshow("Frame", gray);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}