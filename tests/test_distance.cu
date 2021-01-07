#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <knng/config.cuh>
#include <knng/distance.cuh>
#include <knng/utils.cuh>

__global__ void ComputeDistance(float *data, uint32_t a, uint32_t b, int dim, float *result) {
  knng::L2Distance distance(data, data + a * dim, dim);
  float ret = distance.Compare(b);
  if (threadIdx.x == 0) {
    *result = ret;
  }
}

int main(void) {
  const int DIM = 128;
  float data[2 * DIM];
  float result;
  float *cuda_data, *cuda_result;

  for (int i = 0; i < DIM; i++) {
    data[i] = 1.0f;
    data[i + DIM] = 3.0f;
  }

  cudaMalloc(&cuda_data, 2 * DIM * sizeof(float));
  cudaMemcpy(cuda_data, data, 2 * DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&cuda_result, sizeof(float));

  ComputeDistance<<<1, BLOCK_DIM_X>>>(cuda_data, 0, 1, DIM, cuda_result);
  CheckCudaStatus();

  cudaMemcpy(&result, cuda_result, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << result << std::endl;

  cudaFree(cuda_data);
  cudaFree(cuda_result);

  return 0;
}