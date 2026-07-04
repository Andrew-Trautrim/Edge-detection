#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stack>
#include <utility>

#include "Common.cuh"
#include "Kernels.cuh"

Kernel construct_gaussian(int size, float sigma)
{
    float* kernel = (float*)malloc(size * size * sizeof(float));

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

float* gradient_direction(float* x_grad, float* y_grad, int width, int height)
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
    Kernels::gradient_direction(d_x_grad, d_y_grad, width, height, d_result);

    // Copy result from GPU 
    float* result = (float *)std::malloc(width * height * sizeof(float));
    cudaMemcpy(result, d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x_grad);
    cudaFree(d_y_grad);
    cudaFree(d_result);

    return result;
}

float* non_maximum_suppression(float* grad_magnitude, float* grad_direction, int width, int height)
{
    // Allocate GPU memory for gradients and result
    float* d_grad_magnitude;
    float* d_grad_direction;
    float* d_result;

    cudaMalloc(&d_grad_magnitude, width * height * sizeof(float));
    cudaMalloc(&d_grad_direction, width * height * sizeof(float));
    cudaMalloc(&d_result, width * height * sizeof(float));

    // Copy image and kernel to GPU
    cudaMemcpy(d_grad_magnitude, grad_magnitude, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_direction, grad_direction, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Apply kernel to image
    Kernels::non_maximum_suppression(d_grad_magnitude, d_grad_direction, width, height, d_result);

    // Copy result from GPU 
    float* result = (float *)std::malloc(width * height * sizeof(float));
    cudaMemcpy(result, d_result, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_grad_magnitude);
    cudaFree(d_grad_direction);
    cudaFree(d_result);

    return result;
}

unsigned char* double_threshold(float* input, int width, int height, int low, int high)
{
    // Allocate GPU memory for gradients and result
    float* d_input;
    unsigned char* d_result;

    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_result, width * height * sizeof(unsigned char));

    // Copy image and kernel to GPU
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Apply kernel to image
    Kernels::double_threshold(d_input, width, height, low, high, d_result);

    // Copy result from GPU 
    unsigned char* result = (unsigned char*)std::malloc(width * height * sizeof(unsigned char));
    cudaMemcpy(result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
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

        // Clamp incase of floating point shenanigans
        value = std::clamp(value, 0.0f, 255.0f);

        // Round to neared int
        result[i] = static_cast<unsigned char>(value + 0.5f);
    }

    return result;
}

void hysteresis(unsigned char* input, int width, int height)
{
    // Load in all STRONG edges
    std::stack<int> coordinates;
    for (int i = 0; i < width * height; ++i)
    {
        if (input[i] == STRONG)
        {
            coordinates.push(i);
        }
    }

    // DFS change all WEAK edges next to STRONG edges to STRONG edges
    while (!coordinates.empty())
    {
        int idx = coordinates.top();
        coordinates.pop();

        int x = idx % width;
        int y = idx / width;

        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                if (i == j)
                {
                    continue;
                }

                int y_i = y + i;
                int x_j = x + j;
                if (y_i < 0 || y_i >= height || x_j < 0 || x_j >= width)
                {
                    continue;
                }

                int idx = y_i * width + x_j;
                if (input[idx] == WEAK)
                {
                    input[idx] = STRONG;
                    coordinates.push(idx);
                }
            }
        }
    }

    // Remove all remaining WEAK edges
    for (int i = 0; i < width * height; ++i)
    {
        if (input[i] == WEAK)
        {
            input[i] = NONE;
        }
    }
}

float maximum(float* input, size_t size)
{
    float max = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        if (max < input[i])
        {
            max = input[i];
        }
    }

    return max;
}
