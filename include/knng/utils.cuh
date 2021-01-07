#ifndef KNNG_UTILS_CUH_
#define KNNG_UTILS_CUH_

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define CheckCudaStatus()                                                     \
  do {                                                                        \
    cudaError_t cudaerr = cudaDeviceSynchronize();                            \
    if (cudaerr != cudaSuccess)                                               \
      printf("[File: %s line: %d] kernel launch failed with error \"%s\".\n", \
             __FILE__,                                                        \
             __LINE__,                                                        \
             cudaGetErrorString(cudaerr));                                    \
  } while (false)

#endif