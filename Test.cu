#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

#include "Kernels.cuh"

float* convolve(unsigned char* image, int image_width, int image_height, float* kernel, int kernel_width, int kernel_height);
unsigned char* normalize(float* input, size_t size);

float x_grad[3 * 3] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

int main() 
{
    // Read image
    cv::Mat image = cv::imread("../Images/flower.png", cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Failed to load image.\n";
        return -1;
    }

    float* result = convolve(image.data, image.cols, image.rows, x_grad, 3, 3);
    unsigned char* normalized_result = normalize(result, image.cols * image.rows);

    cv::Mat result_image(image.rows, image.cols, image.type(), normalized_result);

    // Write image
    cv::imwrite("output.png", result_image);

    free(result);
    free(normalized_result);
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
