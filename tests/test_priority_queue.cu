#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <knng/config.cuh>
#include <knng/priority_queue.cuh>
#include <knng/utils.cuh>

__global__ void TestPriorityQueue() {
  knng::PriorityQueue queue(32);
  for (int i = 0; i < 100; i++) {
    queue.Add(i, 100.0f - i);
    queue.Add(i, 100.0f - i);
  }
  queue.Print();
  if (threadIdx.x == 0) {
    printf("top: %d\n", queue.ids[0]);
  }
}

int main(void) {
  TestPriorityQueue<<<1, BLOCK_DIM_X>>>();
  CheckCudaStatus();

  return 0;
}
