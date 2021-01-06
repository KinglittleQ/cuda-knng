#include <iostream>

#include <knng/interface.cuh>

int main(void) {
  constexpr int num = 1000000;
  float *data = new float[DIM * num];

  for (int i = 0; i < DIM * num; i++) {
    data[i] = 0.0f;
  }
  data[0] = 1.0f;
  data[DIM * 2] = 1.0f;
  data[DIM * 3] = 1.0f;
  data[DIM * 6] = 1.0f;
  data[DIM * 12] = 1.0f;

  knng::NNDescent index(data, num);
  index.Build(10);

  delete[] data;

  return 0;
}