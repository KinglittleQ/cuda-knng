#include <iostream>

#include <knng/interface.cuh>

int main(void) {
  constexpr int num = 256;
  float data[DIM * num];

  NNDescent index(data, num);
  index.Build(1);

  return 0;
}