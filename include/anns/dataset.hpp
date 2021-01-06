#ifndef ANNS_DATASET_HPP_
#define ANNS_DATASET_HPP_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace anns {

class Dataset {
 public:
  void LoadFloatVecs(const std::string filename, float *&data, size_t &nv, size_t &nd);
  void LoadIntVecs(const std::string filename, int *&data, size_t &nv, size_t &nd);

  void LoadBase(float *&data, size_t &nv, size_t &nd);
  void LoadQuery(float *&data, size_t &nv, size_t &nd);
  void LoadGroundtruth(int *&data, size_t &nv, size_t &nd);

 protected:
  static std::string ConcatPath(const std::string s1, const std::string s2);
  std::string name_;
  std::string data_path_;
  std::string base_filename_;
  std::string query_filename_;
  std::string gt_filename_;
};

void Dataset::LoadFloatVecs(const std::string filename, float *&data, size_t &nv, size_t &nd) {
  std::cout << "Loading data from " << filename << std::endl;
  std::ifstream in(filename, std::ios::binary);

  if (!in.is_open()) {
    std::cout << "Cannot open file " << filename << std::endl;
    exit(1);
    return;
  }

  // Get #dimensions
  int num_dims;
  in.read((char *)&num_dims, 4);
  nd = num_dims;

  // Get file size
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;

  // fisze = (4 + 4d) * n
  nv = fsize / (1 + nd) / 4;

  data = new float[nv * nd];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < nv; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * nd), nd * 4);
  }
  in.close();

  std::cout << "Data loading is done: #dimensions=" << nd << " #vectors=" << nv << std::endl;
}

void Dataset::LoadIntVecs(const std::string filename, int *&data, size_t &nv, size_t &nd) {
  float *ptr;
  LoadFloatVecs(filename, ptr, nv, nd);
  data = reinterpret_cast<int *>(ptr);
}

void Dataset::LoadBase(float *&data, size_t &nv, size_t &nd) {
  LoadFloatVecs(base_filename_, data, nv, nd);
}

void Dataset::LoadQuery(float *&data, size_t &nv, size_t &nd) {
  LoadFloatVecs(query_filename_, data, nv, nd);
}

void Dataset::LoadGroundtruth(int *&data, size_t &nv, size_t &nd) {
  LoadIntVecs(gt_filename_, data, nv, nd);
}

std::string Dataset::ConcatPath(const std::string s1, const std::string s2) {
  if (s1.back() == '/' || s1.back() == '\\') {
    return s1 + s2;
  } else {
    return s1 + "/" + s2;
  }
}

}  // namespace anns

#endif