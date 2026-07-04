
#include "Common.cuh"
#include "EdgeDetection.cuh"

float x_grad_data[3*3] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

float y_grad_data[3*3] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

float laplacian_data[3*3] = {
    0,  1,  0,
    1, -4,  1,
    0,  1,  0
};

Kernel x_grad { 3, 3, x_grad_data };
Kernel y_grad { 3, 3, y_grad_data };
Kernel laplacian_kernel { 3, 3, laplacian_data };

cv::Mat sobel(cv::Mat image, bool reduce_noise)
{
    cv::Mat imagef;
    image.convertTo(imagef, CV_32F);

    float* start_image = reinterpret_cast<float*>(imagef.data);

    // (Optionally) apply Gaussian
    if (reduce_noise)
    {
        Kernel gaussian_kernel = construct_gaussian(7, 1.0f);
        start_image = convolve(
            start_image, 
            image.cols, 
            image.rows, 
            gaussian_kernel.value, 
            gaussian_kernel.width, 
            gaussian_kernel.height);
        free(gaussian_kernel.value);
    }

    // Calculate x and y gradients
    float* x_grad_result = convolve(start_image, image.cols, image.rows, x_grad.value, x_grad.width, x_grad.height);
    float* y_grad_result = convolve(start_image, image.cols, image.rows, y_grad.value, y_grad.width, y_grad.height);
    if (reduce_noise) free(start_image);

    // Calculate gradient magnitude
    float* grad_result = gradient_magnitude(x_grad_result, y_grad_result, image.cols, image.rows);
    free(x_grad_result);
    free(y_grad_result);

    // Normalize result
    unsigned char* normalized_result = normalize(grad_result, image.cols * image.rows);
    free(grad_result);

    // Reconstruct image
    cv::Mat result_image(image.rows, image.cols, image.type());
    memcpy(result_image.data, normalized_result, image.rows * image.cols);
    free(normalized_result);

    // Return result
    return result_image;
}

cv::Mat laplacian(cv::Mat image, bool reduce_noise)
{
    cv::Mat imagef;
    image.convertTo(imagef, CV_32F);

    float* start_image = reinterpret_cast<float*>(imagef.data);

    // (Optionally) apply Gaussian
    if (reduce_noise)
    {
        Kernel gaussian_kernel = construct_gaussian(7, 1.0f);
        start_image = convolve(
            start_image, 
            image.cols, 
            image.rows, 
            gaussian_kernel.value, 
            gaussian_kernel.width, 
            gaussian_kernel.height);
        free(gaussian_kernel.value);
    }

    // Calculate laplacian
    float* laplacian_result = convolve(
        start_image, 
        image.cols, 
        image.rows, 
        laplacian_kernel.value, 
        laplacian_kernel.width, 
        laplacian_kernel.height);
    if (reduce_noise) free(start_image);

    // Normalize result
    unsigned char* normalized_result = normalize(laplacian_result, image.cols * image.rows);
    free(laplacian_result);

    // Reconstruct image
    cv::Mat result_image(image.rows, image.cols, image.type());
    memcpy(result_image.data, normalized_result, image.rows * image.cols);
    free(normalized_result);

    // Return result
    return result_image;
}

cv::Mat canny(cv::Mat image, bool reduce_noise)
{
    cv::Mat imagef;
    image.convertTo(imagef, CV_32F);

    float* start_image = reinterpret_cast<float*>(imagef.data);

    // (Optionally) apply Gaussian
    if (reduce_noise)
    {
        Kernel gaussian_kernel = construct_gaussian(7, 1.0f);
        start_image = convolve(
            start_image, 
            image.cols, 
            image.rows, 
            gaussian_kernel.value, 
            gaussian_kernel.width, 
            gaussian_kernel.height);
        free(gaussian_kernel.value);
    }

    // Calculate x and y gradients
    float* x_grad_result = convolve(start_image, image.cols, image.rows, x_grad.value, x_grad.width, x_grad.height);
    float* y_grad_result = convolve(start_image, image.cols, image.rows, y_grad.value, y_grad.width, y_grad.height);
    if (reduce_noise) free(start_image);

    // Calculate gradient magnitude and direction
    float* grad_magnitude_result = gradient_magnitude(x_grad_result, y_grad_result, image.cols, image.rows);
    float* grad_direction_result = gradient_direction(x_grad_result, y_grad_result, image.cols, image.rows);
    free(x_grad_result);
    free(y_grad_result);

    // Perform non-maximum suppression
    float* supressed_result = non_maximum_suppression(grad_magnitude_result, grad_direction_result, image.cols, image.rows);
    free(grad_magnitude_result);
    free(grad_direction_result);

    // Double Thresholding
    float max = maximum(supressed_result, image.cols * image.rows);
    float low = 0.1f * max;
    float high = 0.2f * max;
    unsigned char* threshold_result = double_threshold(supressed_result, image.cols, image.rows, low, high);
    free(supressed_result);

    // Hysteresis
    hysteresis(threshold_result, image.cols, image.rows);

    // Reconstruct image
    cv::Mat result_image(image.rows, image.cols, image.type());
    memcpy(result_image.data, threshold_result, image.rows * image.cols);
    free(threshold_result);

    // Return result
    return result_image;
}
