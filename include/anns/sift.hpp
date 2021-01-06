#ifndef ANNS_SIFT_HPP_
#define ANNS_SIFT_HPP_

#include "dataset.hpp"

namespace anns {

class SIFTDataset : public Dataset {
 public:
  explicit SIFTDataset(std::string data_path) {
    name_ = "SIFT";
    data_path_ = data_path;
    base_filename_ = ConcatPath(data_path_, "sift_base.fvecs");
    query_filename_ = ConcatPath(data_path_, "sift_query.fvecs");
    gt_filename_ = ConcatPath(data_path_, "sift_groundtruth.ivecs");
  }
};

}  // namespace anns

#endif