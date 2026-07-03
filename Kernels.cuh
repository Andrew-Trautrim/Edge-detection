#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Kernels
{
    __host__ void convolve(
        float* kernel, 
        int kernel_width, 
        int kernel_height, 
        float* image, 
        int image_width,
        int image_height,
        float* result);

    __host__ void gradient_magnitude(
        float* x_grad,
        float* y_grad,
        int width,
        int height,
        float* result);

    __host__ void gradient_direction(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result);

    __host__ void non_maximum_supression(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result);
}