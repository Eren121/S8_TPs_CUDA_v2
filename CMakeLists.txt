cmake_minimum_required(VERSION 3.17)
project(S8_TPs_CUDA_v2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(S8_TPs_CUDA_v2 main.cu)

set_target_properties(
        S8_TPs_CUDA_v2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)