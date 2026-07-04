
struct Kernel
{
    int width;
    int height;
    float* value;
};

Kernel construct_gaussian(int size, float sigma);
float* convolve(float* image, int image_width, int image_height, float* kernel, int kernel_width, int kernel_height);
float* gradient_magnitude(float* x_grad, float* y_grad, int width, int height);
float* gradient_direction(float* x_grad, float* y_grad, int width, int height);
float* non_maximum_suppression(float* grad_magnitude, float* grad_direction, int width, int height);
unsigned char* double_threshold(float* input, int width, int height, int low, int high);
unsigned char* normalize(float* input, size_t size);
void hysteresis(unsigned char* input, int width, int height);
float maximum(float* input, size_t size);