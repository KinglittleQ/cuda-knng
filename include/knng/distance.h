//
// Created by 付聪 on 2017/6/21.
//

#ifndef KNNG_DISTANCE_H
#define KNNG_DISTANCE_H

#include <iostream>
#include <x86intrin.h>
namespace knng {
enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
class Distance {
 public:
  virtual float compare(const float *a, const float *b, unsigned length) const = 0;
  virtual ~Distance() {}
};

class DistanceL2 : public Distance {
 public:
  float compare(const float *a, const float *b, unsigned size) const {
    float result = 0;

    float diff0, diff1, diff2, diff3;
    const float *last = a + size;
    const float *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group) {
      diff0 = a[0] - b[0];
      diff1 = a[1] - b[1];
      diff2 = a[2] - b[2];
      diff3 = a[3] - b[3];
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
      diff0 = *a++ - *b++;
      result += diff0 * diff0;
    }

    return result;
  }
};

}  // namespace efanna2e

#endif  // EFANNA2E_DISTANCE_H
