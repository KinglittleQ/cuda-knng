#ifndef KNNG_DISTANCE_CUH_
#define KNNG_DISTANCE_CUH_

#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <knng/config.cuh>

namespace knng {

struct L2Distance {
  static constexpr int ITEMS_PER_THREAD = (DIM - 1) / BLOCK_DIM_X + 1;

  using BlockReduce = cub::BlockReduce<ValueT, BLOCK_DIM_X>;
  using TempStorage = typename BlockReduce::TempStorage;

  const float *data;
  // float query[DIM];
  const float *query;
  TempStorage *storage;

  __DEVICE__ L2Distance(const float *global_data,
                                        const float *global_query)
      : data(global_data) {
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
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int tid = i * BLOCK_DIM_X + threadIdx.x;
      if (tid < DIM) {
        float diff = query[tid] - data[p * DIM + tid];
        result += diff * diff;
      }
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