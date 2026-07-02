#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Kernels
{
    __host__ void convolve(
        float* kernel, 
        int kernel_width, 
        int kernel_height, 
        unsigned char* image, 
        int image_width,
        int image_height,
        float* result);
}