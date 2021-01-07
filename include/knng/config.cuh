#ifndef KNNG_CONFIG_CUH
#define KNNG_CONFIG_CUH

#include <cstdint>

#define __DEVICE__ __device__ __forceinline__

constexpr int DIM = 128;
constexpr int KG = 32;
constexpr int BLOCK_DIM_X = KG > DIM ? KG : DIM;
// constexpr int MAX_KG = BLOCK_DIM_X;

using ValueT = float;
using KeyT = uint32_t;

#endif