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

cv::Mat sobel(cv::Mat image, bool reduce_noise = false);

float* convolve(float* image, int image_width, int image_height, float* kernel, int kernel_width, int kernel_height);
float* gradient_magnitude(float* x_grad, float* y_grad, int width, int height);

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

Kernel construct_gaussian(int size, float sigma)
{
    float* kernel = new float[size * size];

    const int radius = size / 2;
    const float sigma2 = sigma * sigma;
    const float coefficient = 1.0f / (2.0f * static_cast<float>(M_PI) * sigma2);

    float sum = 0.0f;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            float value = coefficient * std::exp(-(x * x + y * y) / (2.0f * sigma2));
            kernel[(y + radius) * size + (x + radius)] = value;

            sum += value;
        }
    }

    for (int i = 0; i < size*size; ++i)
    {
        kernel[i] /= sum;
    }

    return Kernel { size, size, kernel };
}

float* convolve(
    float* image, 
    int image_width, 
    int image_height, 
    float* kernel, 
    int kernel_width, 
    int kernel_height)
{
    // Allocate GPU memory for image, result, and kernel
    float* d_image;
    float* d_result;
    float* d_kernel;

    cudaMalloc(&d_image, image_width * image_height * sizeof(float));
    cudaMalloc(&d_result, image_width * image_height * sizeof(float));
    cudaMalloc(&d_kernel, kernel_width * kernel_height * sizeof(float));

    // Copy image and kernel to GPU
    cudaMemcpy(d_image, image, image_width * image_height * sizeof(float), cudaMemcpyHostToDevice);
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

float* gradient_magnitude(float* x_grad, float* y_grad, int width, int height)
{
    // Allocate GPU memory for gradients and result
    float* d_x_grad;
    float* d_y_grad;
    float* d_result;

    cudaMalloc(&d_x_grad, width * height * sizeof(float));
    cudaMalloc(&d_y_grad, width * height * sizeof(float));
    cudaMalloc(&d_result, width * height * sizeof(float));

    // Copy image and kernel to GPU
    cudaMemcpy(d_x_grad, x_grad, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_grad, y_grad, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Apply kernel to image
    Kernels::gradient_magnitude(d_x_grad, d_y_grad, width, height, d_result);

    // Copy result from GPU 
    float* result = (float *)std::malloc(width * height * sizeof(float));
    cudaMemcpy(result, d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x_grad);
    cudaFree(d_y_grad);
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
