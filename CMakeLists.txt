CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

project(KNNG)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

set(CUDA_ARCH_LIST Auto)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

set(CUB_INCLUDE_PATH "3rdparty/cub")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xptxas=-v "
                    "--expt-relaxed-constexpr "
                    "--default-stream per-thread")

message("CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(KNNG_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(test_distance tests/test_distance.cu)
target_include_directories(test_distance PRIVATE
                           ${CUB_INCLUDE_PATH}
                           ${KNNG_INCLUDE_PATH})
target_link_libraries(test_distance ${CUDA_curand_LIBRARY})

cuda_add_executable(test_priority_queue tests/test_priority_queue.cu)
target_include_directories(test_priority_queue PRIVATE
                           ${KNNG_INCLUDE_PATH})
target_link_libraries(test_priority_queue ${CUDA_curand_LIBRARY})

cuda_add_executable(test_knng tests/test_knng.cu)
target_include_directories(test_knng PRIVATE
                           ${CUB_INCLUDE_PATH}
                           ${KNNG_INCLUDE_PATH})
target_link_libraries(test_knng ${CUDA_curand_LIBRARY})
