#ifndef KNNG_INTERFACE_HH_
#define KNNG_INTERFACE_HH_

#include <cstring>
#include <limits>
#include <memory>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#include <knng/config.cuh>
#include <knng/utils.cuh>
#include <knng/graph.cuh>


struct NNDescent {
  const float *data;
  size_t num;
  uint32_t *graph;
  KNNGraph *knn_graph;

  float *cuda_data;
  uint32_t *cuda_graph, *cuda_new_graph;

  NNDescent(const float *data, size_t num) : data(data), num(num) {
    graph = new uint32_t[K * num];
    cudaMalloc(&cuda_data, num * DIM * sizeof(float));
    cudaMemcpy(cuda_data, data, num * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_graph, K * num * sizeof(uint32_t));
    cudaMalloc(&cuda_new_graph, K * num * sizeof(uint32_t));

    KNNGraph tmp(cuda_data, cuda_graph, cuda_new_graph, num);
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
    // knn_graph->InitGraph<<<num_blocks, BLOCK_DIM_X>>>();
    InitGraph<<<num_blocks, BLOCK_DIM_X>>>(knn_graph);

    for (int i = 0; i < num_iters; i++) {
      RefineGraph<<<num, BLOCK_DIM_X>>>(knn_graph, i);
      CheckCudaStatus();
      SwapGraph<<<num, K>>>(knn_graph);
      CheckCudaStatus();
    }

    cudaMemcpy(graph, cuda_graph, num * K * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CheckCudaStatus();
  }
};



#endif