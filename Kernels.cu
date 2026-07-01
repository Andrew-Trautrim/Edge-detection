#include "Kernels.cuh"

const int THREADS_PER_DIM = 16;

namespace Kernels
{
    __global__ void convolve_kernel(
        double* kernel, 
        int kernel_width, 
        int kernel_height, 
        double* image, 
        int image_width,
        int image_height,
        double* result);

    __host__ void convolve(
        double* kernel, 
        int kernel_width, 
        int kernel_height, 
        double* image, 
        int image_width,
        int image_height,
        double* result)
    {
        // Set kernel parameters
        int blocks_y = (image_height + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (image_width + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 BLOCKS(blocks_x, blocks_y);
        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);

        convolve_kernel<<<BLOCKS, THREADS>>>(
            kernel,
            kernel_width,
            kernel_height,
            image,
            image_width,
            image_height,
            result
        );

        cudaDeviceSynchronize();
    }

    __global__ void convolve_kernel(
        double* kernel, 
        int kernel_width, 
        int kernel_height, 
        double* image, 
        int image_width,
        int image_height,
        double* result)
    {
        // Calculate row + col for each thread
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= image_width || y >= image_height) {
            return;
        }

        // Get radius of kernel
        int y_rad = kernel_height / 2;
        int x_rad = kernel_width / 2;

        double sum = 0.0;

        for (int k_y = -1*y_rad; k_y <= y_rad; ++k_y) 
        {
            for (int k_x = -1*x_rad; k_x <= x_rad; ++k_x) 
            {
                int i_y = y + k_y;
                int i_x = x + k_x;

                // ignore part of kernel that doesnt overlap with image
                if (i_y >=0 && i_y < image_height
                    && i_x >= 0 && i_x < image_width)
                {
                    // flip kernel and apply to image
                    sum += kernel[(y_rad - k_y) * kernel_width + (x_rad - k_x)] * image[i_y * image_width + i_x];
                }
            }
        }

        result[y * image_width + x] = sum;
    }
}