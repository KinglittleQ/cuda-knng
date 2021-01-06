CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

project(KNNG)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

set(CUDA_ARCH_LIST Auto)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

# set(EIGEN_INCLUDE_PATH "3rdparty/eigen-git-mirror")
set(CUB_INCLUDE_PATH "3rdparty/cub")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xptxas=-v --expt-relaxed-constexpr --default-stream per-thread" )

message("CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(DATASET_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/datasets)

cuda_add_executable(test_distance unit_tests/test_distance.cu)
target_include_directories(test_distance
                           PRIVATE
                           ${CUDA_NVCC_FLAGS}
                           ${CUB_INCLUDE_PATH}
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_distance ${CUDA_curand_LIBRARY})

cuda_add_executable(test_priority_queue unit_tests/test_priority_queue.cu)
target_include_directories(test_priority_queue
                           PRIVATE
                           ${CUDA_NVCC_FLAGS}
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_priority_queue ${CUDA_curand_LIBRARY})

cuda_add_executable(test_knng unit_tests/test_knng.cu)
target_include_directories(test_knng
                           PRIVATE
                           ${CUDA_NVCC_FLAGS}
                           ${CUB_INCLUDE_PATH}
                           ${DATASET_INCLUDE_PATH}
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_knng ${CUDA_curand_LIBRARY})
