#ifndef KNNG_CONFIG_CUH
#define KNNG_CONFIG_CUH

#include <cstdint>

#define __DEVICE__ __device__ __forceinline__

constexpr int BLOCK_DIM_X = 128;
constexpr int DIM = 128;
constexpr int K = 128;

using ValueT = float;
using KeyT = uint32_t;


#endif