#include <opencv2/opencv.hpp>

cv::Mat sobel(cv::Mat image, bool reduce_noise = false);
cv::Mat laplacian(cv::Mat image, bool reduce_noise = false);
cv::Mat canny(cv::Mat image, bool reduce_noise = false);