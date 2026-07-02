cmake_minimum_required(VERSION 3.18)

project(EdgeDetection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(EdgeDetection
    Test.cu
    Kernels.cu
)

target_include_directories(EdgeDetection PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(EdgeDetection PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
)