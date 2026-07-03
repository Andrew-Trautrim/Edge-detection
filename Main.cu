#include <opencv2/opencv.hpp>
#include <iostream>

#include "EdgeDetection.cuh"

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

    cv::Mat sobel_result = sobel(image, true);
    cv::Mat laplacian_result = laplacian(image, true);
    cv::Mat canny_result = canny(image, true);

    // Write images
    cv::imwrite("../Output/sobel.png", sobel_result);
    cv::imwrite("../Output/laplacian.png", laplacian_result);
    cv::imwrite("../Output/canny.png", laplacian_result);
}