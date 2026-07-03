#include "Kernels.cuh"

const int THREADS_PER_DIM = 16;

namespace Kernels
{
    __global__ void convolve_kernel(
        float* kernel, 
        int kernel_width, 
        int kernel_height, 
        float* image, 
        int image_width,
        int image_height,
        float* result);

    __global__ void gradient_magnitude_kernel(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result);
        
    __global__ void gradient_direction_kernel(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result);

    __global__ void non_maximum_supression_kernel(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result);

    __host__ void convolve(
        float* kernel, 
        int kernel_width, 
        int kernel_height, 
        float* image, 
        int image_width,
        int image_height,
        float* result)
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

    __host__ void gradient_magnitude(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result)
    {
        // Set kernel parameters
        int blocks_y = (height + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (width + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 BLOCKS(blocks_x, blocks_y);
        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);

        gradient_magnitude_kernel<<<BLOCKS, THREADS>>>(x_grad, y_grad, width, height, result);

        cudaDeviceSynchronize();
    }

    __host__ void gradient_direction(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result)
    {
        // Set kernel parameters
        int blocks_y = (height + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (width + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 BLOCKS(blocks_x, blocks_y);
        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);

        gradient_direction_kernel<<<BLOCKS, THREADS>>>(x_grad, y_grad, width, height, result);

        cudaDeviceSynchronize();
    }

    __host__ void non_maximum_supression(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result)
    {
        // Set kernel parameters
        int blocks_y = (height + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (width + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 BLOCKS(blocks_x, blocks_y);
        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);

        non_maximum_supression_kernel<<<BLOCKS, THREADS>>>(grad_magnitude, grad_direction, width, height, result);

        cudaDeviceSynchronize();
    }

    __global__ void convolve_kernel(
        float* kernel, 
        int kernel_width, 
        int kernel_height, 
        float* image, 
        int image_width,
        int image_height,
        float* result)
    {
        // Calculate row + col for each thread
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= image_width || y >= image_height) 
        {
            return;
        }

        // Get radius of kernel
        int y_rad = kernel_height / 2;
        int x_rad = kernel_width / 2;

        float sum = 0.0f;

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

    __global__ void gradient_magnitude_kernel(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result)
    {
        // Calculate row + col for each thread
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= width || y >= height) 
        {
            return;
        }

        int idx = y * width + x;
        result[idx] = sqrtf((x_grad[idx] * x_grad[idx]) + (y_grad[idx] * y_grad[idx]));
    }

    __global__ void gradient_direction_kernel(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result)
    {
        // Calculate row + col for each thread
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= width || y >= height) 
        {
            return;
        }

        int idx = y * width + x;
        result[idx] = atan2f(y_grad[idx], x_grad[idx]);
    }

    __global__ void non_maximum_supression_kernel(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result)
    {
        // Calculate row + col for each thread
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= width || y >= height) 
        {
            return;
        }

        int idx = y * width + x;

        // Ignore edges
        if (x == 0 || x == width - 1 
            || y == 0 || y == height - 1) 
        {
            result[idx] = 0.0f;
            return;
        }

        float angle = grad_direction[idx] * 180.0f / M_PI;
        if (angle < 0.0f)
        {
            angle += 180.0f;
        }

        float n1, n2;
        if (angle < 22.5f || angle >= 157.5f)
        {
            // 0° - check left and right
            n1 = grad_magnitude[idx - 1]; // y * width + x - 1 = idx - 1
            n2 = grad_magnitude[idx + 1]; // y * width + x + 1 = idx + 1
        }
        else if (angle < 67.5f)
        {
            // 45° - check upper right and lower left
            n1 = grad_magnitude[(y - 1) * width + (x + 1)];
            n2 = grad_magnitude[(y + 1) * width + (x - 1)];
        }
        else if (angle < 112.5f)
        {
            // 90° - check up and down
            n2 = grad_magnitude[(y - 1) * width + x];
            n1 = grad_magnitude[(y + 1) * width + x];
        }
        else
        {
            // 135° - check upper left and lower right
            n1 = grad_magnitude[(y - 1) * width + (x - 1)];
            n2 = grad_magnitude[(y + 1) * width + (x + 1)];
        }

        float curr = grad_magnitude[idx];
        if (curr >= n1 && curr >= n2)
        {
            result[idx] = curr;
        }
        else 
        {
            result[idx] = 0.0f;
        }
    }
}