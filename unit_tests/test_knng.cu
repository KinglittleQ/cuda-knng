#include <iostream>

#include <knng/interface.cuh>
#include <sift.hpp>


double ComputeRecall(uint32_t num,
                     int K,
                     const int *gt,
                     const std::vector<std::vector<unsigned>> &points);

int main(int argc, char **argv) {
  float *base_data, *query_data;
  int *gt_data;
  size_t nb, nq, d, topk;
  
  anns::SIFTDataset dataset(argv[1]);
  dataset.LoadBase(base_data, nb, d);
  dataset.LoadQuery(query_data, nq, d);
  dataset.LoadGroundtruth(gt_data, nq, topk);
  
  unsigned L = atoi(argv[2]);
  unsigned iters = atoi(argv[3]);

  knng::NNDescent index(base_data, nb);
  index.Build(iters);
  
  std::vector<std::vector<unsigned>> result;
  result.resize(nq);
  for (size_t i = 0; i < nq; i++) {
    result[i].resize(topk);
    index.Search(query_data + i * d, topk, L, result[i].data());
  }
  double recall = ComputeRecall(nq, topk, gt_data, result);

  printf("Recall@100: %lf\n", recall);

  delete[] base_data;
  delete[] query_data;
  delete[] gt_data;

  return 0;
}

double ComputeRecall(uint32_t num,
                     int K,
                     const int *gt,
                     const std::vector<std::vector<unsigned>> &points) {
  int recalls = 0;
  for (size_t i = 0; i < num; i++) {
    std::set<uint32_t> label;
    for (int j = 0; j < K; j++) {
      label.insert(gt[i * K + j]);
    }
    for (const auto &p : points[i]) {
      if (label.count(p) != 0) {
        recalls += 1;
      }
    }
  }

  return 1.0 * recalls / (num * K);
}