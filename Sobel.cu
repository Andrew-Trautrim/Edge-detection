
#include "Common.cuh"
#include "Sobel.cuh"

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

Kernel x_grad { 3, 3, x_grad_data };
Kernel y_grad { 3, 3, y_grad_data };

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
        delete[] gaussian_kernel.value;
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