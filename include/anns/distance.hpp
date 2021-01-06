#ifndef __ANNS_DISTANCE_HPP__
#define __ANNS_DISTANCE_HPP__

#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                  \
  tmp2 = _mm256_loadu_ps(addr2);                  \
  tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
  tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
  dest = _mm256_add_ps(dest, tmp1);

namespace anns {

class Distance {
 public:
  Distance(size_t dim) : dim_(dim) {}
  virtual float Compare(const float *p1, const float *p2) const = 0;
  virtual ~Distance() = default;

  mutable size_t num = 0;

 protected:
  const size_t dim_;
};

class L2Distance : public Distance {
 public:
  L2Distance(size_t dim) : Distance(dim) {}

  float Compare(const float *p1, const float *p2) const {
    num += 1;
#ifndef __AVX__
    return Sqr_(p1, p2, dim_);
#else
    float result = 0;

    __m256 sum;
    __m256 tmp0, tmp1;
    unsigned residual_size = dim_ % 16;
    unsigned aligned_size = dim_ - residual_size;
    const float *residual_start1 = p1 + aligned_size;
    const float *residual_start2 = p2 + aligned_size;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);

    const float *ptr1 = p1;
    const float *ptr2 = p2;
    for (unsigned i = 0; i < aligned_size; i += 16, ptr1 += 16, ptr2 += 16) {
      AVX_L2SQR(ptr1, ptr2, sum, tmp0, tmp1);
      AVX_L2SQR(ptr1 + 8, ptr2 + 8, sum, tmp0, tmp1);
    }
    if (residual_size >= 8) {
      AVX_L2SQR(residual_start1, residual_start2, sum, tmp0, tmp1);
      residual_size -= 8;
      residual_start1 += 8;
      residual_start2 += 8;
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] +
             unpack[7];

    if (residual_size > 0) {
      result += Sqr_(residual_start1, residual_start2, residual_size);
    }
    return result;
#endif
  }

 private:
  float Sqr_(const float *p1, const float *p2, uint32_t size) const {
    // Auto vectorized with -O3 flag
    float sum = 0;
    for (size_t i = 0; i < size; i++) {
      float tmp = p1[i] - p2[i];
      sum += tmp * tmp;
    }
    return sum;
  }
};

class L1Distance : public Distance {
 public:
  L1Distance(size_t dim) : Distance(dim) {}
  float operator()(const float *p1, const float *p2) const {
    num += 1;
    float sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      sum += fabs(p1[i] - p2[i]);
    }
    return sum;
  }
};

}  // namespace anns

#endif