#ifndef KNNG_INTERFACE_CUH_
#define KNNG_INTERFACE_CUH_

#include <algorithm>
#include <anns/distance.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <knng/config.cuh>
#include <knng/graph.cuh>
#include <knng/utils.cuh>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace knng {

void Print(uint32_t *graph, int KG) {
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < KG; j++) {
      printf("%u ", graph[i * KG + j]);
    }
    printf("\n");
  }
}

struct Neighbor {
  unsigned id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

  inline bool operator<(const Neighbor &other) const { return distance < other.distance; }
};

void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

inline unsigned InsertIntoPool(Neighbor *addr, unsigned M, Neighbor nn) {
  // find the location to insert
  unsigned left = 0, right = M - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], M * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[M] = nn;
    return M;
  }
  while (right > 1 && left < right - 1) {
    unsigned mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance)
      right = mid;
    else
      left = mid;
  }
  // checM equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return M + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) return M + 1;
  memmove((char *)&addr[right + 1], &addr[right], (M - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

struct NNDescent {
  const float *data;
  size_t num;
  int KG;
  int dim;
  uint32_t *graph;
  KNNGraph *knn_graph;

  float *cuda_data;
  uint32_t *cuda_graph, *cuda_new_graph;
  anns::L2Distance distance;

  NNDescent(const float *data, size_t num, int KG, int dim)
      : data(data), num(num), KG(KG), dim(dim), distance(dim) {
    graph = new uint32_t[KG * num];
    cudaMalloc(&cuda_data, num * dim * sizeof(float));
    cudaMemcpy(cuda_data, data, num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_graph, KG * num * sizeof(uint32_t));
    cudaMalloc(&cuda_new_graph, KG * num * sizeof(uint32_t));

    KNNGraph tmp(cuda_data, cuda_graph, cuda_new_graph, num, KG, dim);
    cudaMalloc(&knn_graph, sizeof(KNNGraph));
    cudaMemcpy(knn_graph, &tmp, sizeof(KNNGraph), cudaMemcpyHostToDevice);
  }

  ~NNDescent() {
    delete[] graph;
    cudaFree(cuda_data);
    cudaFree(cuda_graph);
    cudaFree(cuda_new_graph);
    cudaFree(knn_graph);
  }

  void Build(int num_iters) {
    printf("Init graph ...\n");
    int num_blocks = (num - 1) / BLOCK_DIM_X + 1;
    InitGraph<<<num_blocks, BLOCK_DIM_X>>>(knn_graph);

    for (int i = 0; i < num_iters; i++) {
      RefineGraph<<<num, BLOCK_DIM_X>>>(knn_graph, i);
      CheckCudaStatus();
      SwapGraph<<<num, KG>>>(knn_graph);
      CheckCudaStatus();
    }
    cudaMemcpy(graph, cuda_graph, num * KG * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // Print(graph, KG);
  }

  void Search(const float *query, unsigned topk, unsigned L, unsigned *indices) {
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned)num);

    boost::dynamic_bitset<> flags;
    flags.resize(num);

    for (unsigned i = 0; i < L; i++) {
      unsigned id = init_ids[i];
      float dist = distance.Compare(data + dim * id, query);
      retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < KG; ++m) {
          unsigned id = graph[n * KG + m];
          if (flags.test(id)) continue;
          flags.set(id);
          float dist = distance.Compare(query, data + dim * id);
          if (dist >= retset[L - 1].distance) continue;
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk) nk = r;
        }
        // lock to here
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < topk; i++) {
      indices[i] = retset[i].id;
    }
  }
};

}  // namespace knng

#endif