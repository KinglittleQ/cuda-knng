#ifndef KNNG_GRAPH_CUH_
#define KNNG_GRAPH_CUH_

#include <cstring>
#include <limits>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <knng/config.cuh>
#include <knng/distance.cuh>
#include <knng/priority_queue.cuh>
#include <knng/utils.cuh>

namespace knng {

struct KNNGraph {
  const float *data;
  uint32_t *graph;
  uint32_t *new_graph;
  size_t num;

  KNNGraph(const float *data, uint32_t *graph,
                      uint32_t *new_graph, size_t num)
       : data(data), graph(graph), new_graph(new_graph), num(num) {}
};  

__DEVICE__ void Print(KNNGraph *knn_graph) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < knn_graph->num; i++) {
      for (int j = 0; j < K; j++) {
        printf("%u ", knn_graph->graph[i * K + j]);
      }
      printf("\n");
    }
  }
}

// Init one node per thread
__global__ void InitGraph(KNNGraph *knn_graph) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= knn_graph->num) {
    return;
  }

  // curandGenerator_t gen;
  // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // curandGenerateUniform(gen, d_rng, m_ggnn_graph.getNs(layer));

  // std::mt19937 gen(rd());
  // std::uniform_int_distribution<> distrib(0, knn_graph->num - 1);
  
  for (int i = 0; i < K; i++) {
    int n;
    bool is_duplicate;
    do {
      is_duplicate = false;
      // n = distrib(gen);
      n = (tid + 1000 * i + 1) % knn_graph->num;
      if (n == tid) {
        is_duplicate = true;
      } else {
        for (int j = 0; j < i; j++) {
          if (knn_graph->graph[tid * K + j] == n) {
            is_duplicate = true;
            break;
          }
        }
      }
    } while (is_duplicate);

    knn_graph->graph[tid * K + i] = n;
  }
}

__global__ void SwapGraph(KNNGraph *knn_graph) {
  uint32_t id = blockIdx.x;
  uint32_t n = threadIdx.x;
  if (id >= knn_graph->num || n >= K) {
    return;
  }

  knn_graph->graph[id * K + n] = knn_graph->new_graph[id * K + n];
}

// Refine one node per block
__DEVICE__ void Refine(KNNGraph *knn_graph) {
  uint32_t id = blockIdx.x;
  if (id >= knn_graph->num) {
    return;
  }

  PriorityQueue queue;
  L2Distance distance(knn_graph->data, knn_graph->data + id * DIM);

  for (int i = 0; i < K; i++) {
    uint32_t n = knn_graph->graph[id * K + i];  // neighbor
    float dist = distance.Compare(n);
    queue.Add(n, dist);

    for (int j = 0; j < K; j++) {
      uint32_t nn = knn_graph->graph[n * K + j];  // neighbor's neighbor
      if (nn != id) {
        dist = distance.Compare(nn);
        queue.Add(nn, dist);
        __syncthreads();
      }
    }
  }

  __syncthreads();
  uint32_t k = threadIdx.x;
  if (k >= K) {
    return;
  }

  uint32_t n = queue.ids[k];
  knn_graph->new_graph[id * K + k] = n;
}

__global__ void RefineGraph(KNNGraph *knn_graph, int iter) {
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Iter %d, node %d\n", iter, blockIdx.x);
  }
  Refine(knn_graph);
}

}  // end knng

#endif
