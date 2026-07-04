#include <cuda_runtime.h>
#include <curand_kernel.h>

const unsigned char NONE   = 0;
const unsigned char WEAK   = 128;
const unsigned char STRONG = 255;

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

    __host__ void non_maximum_suppression(
        float* grad_magnitude,
        float* grad_direction,
        int width,
        int height,
        float* result);

    __host__ void double_threshold(
        float* input,
        int width,
        int height,
        float low,
        float high,
        unsigned char* result);
}