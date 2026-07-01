#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Kernels
{
    __host__ void convolve(
        double* kernel, 
        int kernel_width, 
        int kernel_height, 
        double* image, 
        int image_width,
        int image_height,
        double* result);
}