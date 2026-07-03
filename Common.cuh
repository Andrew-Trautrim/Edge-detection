
struct Kernel
{
    int width;
    int height;
    float* value;
};

Kernel construct_gaussian(int size, float sigma);
float* convolve(float* image, int image_width, int image_height, float* kernel, int kernel_width, int kernel_height);
float* gradient_magnitude(float* x_grad, float* y_grad, int width, int height);
unsigned char* normalize(float* input, size_t size);