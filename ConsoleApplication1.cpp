#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Read the image file
    Mat img(256, 256, CV_8UC3);
    randu(img, Scalar(0, 0, 0), Scalar(255, 255, 255));
    

    // Check for failure
    if (img.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    // intensity 
    Vec3b intensity = img.at<Vec3b>(152, 152);
    int blue = intensity.val[0];
    int green = intensity.val[1];
    int red = intensity.val[2];

    cout << blue;
    cout << green;
    cout << red;
    //change the color image to grayscale image
    cvtColor(img, img, COLOR_BGR2GRAY);

    //equalize the histogram
    Mat hist_equalized_image;
    equalizeHist(img, hist_equalized_image);

    //Define names of windows
    String windowNameOfOriginalImage = "Original Image";
    String windowNameOfHistogramEqualized = "Histogram Equalized Image";

    // Show images inside created windows.
    imshow(windowNameOfOriginalImage, img);
    imshow(windowNameOfHistogramEqualized, hist_equalized_image);
    

    waitKey(0); // Wait for any keystroke in one of the windows

    destroyAllWindows(); //Destroy all open windows

    return 0;
}