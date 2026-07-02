#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

#include "Kernels.cuh"

struct Kernel
{
    int width;
    int height;
    float* value;
};

Kernel construct_gaussian(int size, float sigma);
cv::Mat apply_kernel(cv::Mat image, Kernel kernel);
float* convolve(unsigned char* image, int image_width, int image_height, float* kernel, int kernel_width, int kernel_height);
unsigned char* normalize(float* input, size_t size);

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

int main() 
{
    // Read image
    cv::Mat image = cv::imread("../Images/flower.png", cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    Kernel gaussian = construct_gaussian(31, 5.0f);
    cv::Mat result = apply_kernel(image, gaussian);
    cv::imwrite("../Output/gaussian.png", result);

    delete gaussian.value;

    // cv::Mat x_grad_image = apply_kernel(image, x_grad);
    // cv::Mat y_grad_image = apply_kernel(image, y_grad);

    // cv::imwrite("../Output/x_grad.png", x_grad_image);
    // cv::imwrite("../Output/y_grad.png", y_grad_image);
}

// cv::Mat sobel(cv::Mat image)
// {
//     // 1. Calculate a and y gradients
//     float* x_grad_result = convolve(image.data, image.cols, image.rows, x_grad.value, x_grad.width, x_grad.height);
//     float* y_grad_result = convolve(image.data, image.cols, image.rows, y_grad.value, y_grad.width, y_grad.height);

//     // 2. Calculate gradient magnitude

//     // 3. Normalize result
//     unsigned char* normalized_result = normalize(result, image.cols * image.rows);
//     cv::Mat result_image(image.rows, image.cols, image.type(), normalized_result);
// }

Kernel construct_gaussian(int size, float sigma)
{
    float* kernel = new float[size * size];

    const int radius = size / 2;
    const float sigma2 = sigma * sigma;
    const float coefficient = 1.0f / (2.0f * static_cast<float>(M_PI) * sigma2);

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            float value = coefficient * std::exp(-(x * x + y * y) / (2.0f * sigma2));
            kernel[(y + radius) * size + (x + radius)] = value;
        }
    }

    return Kernel { size, size, kernel };
}

cv::Mat apply_kernel(cv::Mat image, Kernel kernel)
{
    float* result = convolve(image.data, image.cols, image.rows, kernel.value, kernel.width, kernel.height);
    unsigned char* normalized_result = normalize(result, image.cols * image.rows);
    cv::Mat result_image(image.rows, image.cols, image.type(), normalized_result);

    free(result);

    return result_image;
}

float* convolve(
    unsigned char* image, 
    int image_width, 
    int image_height, 
    float* kernel, 
    int kernel_width, 
    int kernel_height)
{
    // Allocate GPU memory for image, result, and kernel
    unsigned char* d_image;
    float* d_result;
    float* d_kernel;

    cudaMalloc(&d_image, image_width * image_height * sizeof(unsigned char));
    cudaMalloc(&d_result, image_width * image_height * sizeof(float));
    cudaMalloc(&d_kernel, kernel_width * kernel_height * sizeof(float));

    // Copy image and kernel to GPU
    cudaMemcpy(d_image, image, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_width * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    // Apply kernel to image
    Kernels::convolve(d_kernel, kernel_width, kernel_height, d_image, image_width, image_height, d_result);

    // Copy result from GPU 
    float* result = (float *)std::malloc(image_width * image_height * sizeof(float));
    cudaMemcpy(result, d_result, image_width * image_height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_result);

    return result;
}

unsigned char* normalize(float* input, size_t size)
{
    unsigned char* result = (unsigned char*)std::malloc(size * sizeof(unsigned char));

    float min = input[0];
    float max = input[0];

    for (int i = 1; i < size; ++i)
    {
        if (input[i] < min)
        {
            min = input[i];
        }

        if (input[i] > max)
        {
            max = input[i];
        }
    }

    // Handle constant array
    if (min == max)
    {
        std::fill(result, result + size, 0);
        return result;
    }

    float scale = 255.0f / (max - min);
    for (int i = 0; i < size; ++i)
    {
        // Scale value
        float value = (input[i] - min) * scale;

        // Clamp incase of floting point shenanigans
        value = std::clamp(value, 0.0f, 255.0f);

        // Round to neared int
        result[i] = static_cast<unsigned char>(value + 0.5f);
    }

    return result;
}
