#include <opencv2/opencv.hpp>
#include <iostream>

#include "Sobel.cuh"

int main(int argc, char* argv[]) 
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./EdgeDetection <image>.\n";
        return -1;
    }

    // Read image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    // Determine edges using Sobel operator
    cv::Mat result = sobel(image, true);

    // Write image
    cv::imwrite("../Output/result.png", result);
}