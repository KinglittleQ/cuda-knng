#ifndef KNNG_DISTANCE_CUH_
#define KNNG_DISTANCE_CUH_

#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <knng/config.cuh>

namespace knng {

struct L2Distance {

  using BlockReduce = cub::BlockReduce<ValueT, BLOCK_DIM_X>;
  using TempStorage = typename BlockReduce::TempStorage;

  const float *data;
  const float *query;
  TempStorage *storage;
  int dim;

  __DEVICE__ L2Distance(const float *global_data,
                        const float *global_query, int dim)
         : data(global_data), dim(dim) {
    // copy from global memory to block memory
    // memcpy(query, global_query, sizeof(float) * DIM);
    query = global_query;

    // allocate shared memory
    __shared__ TempStorage shared_storage;
    storage = &shared_storage;
  }
  
  __DEVICE__ float Compare(const uint32_t p) {
    __shared__ float shared_sum;

    float result = 0;
    int tid = threadIdx.x;
    if (tid < dim) {
      float diff = query[tid] - data[p * dim + tid];
      result += diff * diff;
    }

    float sum = BlockReduce(*storage).Sum(result);
    if (threadIdx.x == 0) {
      shared_sum = sum;
    }
    __syncthreads();

    return shared_sum;
  }

};

}  // end knng

#endif