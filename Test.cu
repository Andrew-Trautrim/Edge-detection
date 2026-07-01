#include <iostream>
#include <iomanip>
#include <vector>

#include "Kernels.cuh"

void print_image_vector(std::vector<double> image, int image_width, int image_height);
std::vector<double> convolve(std::vector<double> image, int image_width, int image_height, double* kernel, int kernel_width, int kernel_height);

double x_grad[3 * 3] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

const int image_size = 20;

int main() 
{
    std::vector<double> image(image_size * image_size);
    for (int i = 0; i < image_size * image_size; ++i)
    {
        image[i] = i % 4 == 0 ? 150 : 0;
    }

    std::vector<double> result = convolve(image, image_size, image_size, x_grad, 3, 3);

    std::cout << "Image: " << std::endl;
    print_image_vector(image, image_size, image_size);

    std::cout << "Result: " << std::endl;
    print_image_vector(result, image_size, image_size);
}

void print_image_vector(std::vector<double> image, int image_width, int image_height)
{
    for (int i = 0; i < image_width * image_height; ++i)
    {
        std::cout << std::setw(4) << (int)image[i] << ", ";

        if ((i + 1) % image_width == 0)
        {
            std::cout << std::endl;
        }
    }
}

std::vector<double> convolve(
    std::vector<double> image, 
    int image_width, 
    int image_height, 
    double* kernel, 
    int kernel_width, 
    int kernel_height)
{
    // Allocate GPU memory for image, result, and kernel
    double* d_image;
    double* d_result;
    double* d_kernel;

    cudaMalloc(&d_image, image_width * image_height * sizeof(double));
    cudaMalloc(&d_result, image_width * image_height * sizeof(double));
    cudaMalloc(&d_kernel, kernel_width * kernel_height * sizeof(double));

    // Copy image and kernel to GPU
    cudaMemcpy(d_image, image.data(), image_width * image_height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_width * kernel_height * sizeof(double), cudaMemcpyHostToDevice);

    // Apply kernel to image
    Kernels::convolve(d_kernel, kernel_width, kernel_height, d_image, image_width, image_height, d_result);

    // Copy result from GPU 
    std::vector<double> result(image_width * image_height);
    cudaMemcpy(result.data(), d_result, image_width * image_height * sizeof(double), cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_result);

    return result;
}