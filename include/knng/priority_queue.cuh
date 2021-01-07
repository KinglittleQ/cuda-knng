#ifndef KNNG_PRIORITY_QUEUE_CUH_
#define KNNG_PRIORITY_QUEUE_CUH_

#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <knng/config.cuh>
#include <limits>

namespace knng {

struct PriorityQueue {
  uint32_t *ids;
  float *dists;

  __DEVICE__ PriorityQueue() {
    __shared__ float shared_dists[KG];
    __shared__ uint32_t shared_ids[KG];
    dists = reinterpret_cast<float *>(shared_dists);
    ids = reinterpret_cast<uint32_t *>(shared_ids);

    if (threadIdx.x < KG) {
      dists[threadIdx.x] = std::numeric_limits<float>::infinity();
      ids[threadIdx.x] = std::numeric_limits<uint32_t>::max();
    }
    __syncthreads();
  }

  __DEVICE__ void Add(uint32_t id, float dist) {
    __shared__ bool exists;

    int tid = threadIdx.x;

    if (tid >= KG) {
      return;
    } else if (tid == 0) {
      exists = false;
    }
    __syncthreads();

    uint32_t entry_id = ids[tid];
    float entry_dist = dists[tid];
    if (entry_id == id) {
      exists = true;
    }
    __syncthreads();

    if (exists) {
      return;
    }

    if (entry_dist > dist) {
      if (tid < KG - 1) {
        ids[tid + 1] = entry_id;
        dists[tid + 1] = entry_dist;
      }

      if (tid == 0 || dists[tid - 1] <= dist) {
        ids[tid] = id;
        dists[tid] = dist;
      }
    }
  }

  __DEVICE__ uint32_t Top() { return ids[0]; }

  __DEVICE__ void Print() {
    __syncthreads();
    if (threadIdx.x == 0) {
      printf("Priority queue: \n");
      for (int i = 0; i < KG; i++) {
        printf("%d -> (%d, %lf)\n", i, ids[i], dists[i]);
      }
    }
  }
};

}  // namespace knng

#endif