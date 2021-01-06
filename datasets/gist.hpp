#ifndef ANNS_DATASET_GIST_HPP_
#define ANNS_DATASET_GIST_HPP_

#include "dataset.hpp"

namespace anns {

class GISTDataset : public Dataset {
 public:
  explicit GISTDataset(std::string data_path) {
    name_ = "GIST";
    data_path_ = data_path;
    base_filename_ = ConcatPath(data_path_, "gist_base.fvecs");
    query_filename_ = ConcatPath(data_path_, "gist_query.fvecs");
    gt_filename_ = ConcatPath(data_path_, "gist_groundtruth.ivecs");
  }
};

}  // namespace anns

#endif