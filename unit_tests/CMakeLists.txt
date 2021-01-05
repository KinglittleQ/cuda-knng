
cuda_add_executable(test_distance test_distance.cu)
target_include_directories(
    test_distance
    PRIVATE
    ${CUDA_NVCC_FLAGS}
    # ${EIGEN_INCLUDE_PATH}
    ${CUB_INCLUDE_PATH}
    ${CMAKE_PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_distance ${CUDA_curand_LIBRARY})
