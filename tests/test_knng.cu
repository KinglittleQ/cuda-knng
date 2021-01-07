#include <chrono>
#include <iostream>
#include <set>

#include <anns/sift.hpp>

#include <knng/interface.cuh>

using std::cout;
using std::endl;
using namespace std::chrono;

double ComputeRecall(uint32_t num, int K, const int *gt,
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

int main(int argc, char **argv) {
  float *base_data, *query_data;
  int *gt_data;
  size_t nb, nq, d, topk;

  if (argc != 5) {
    cout << "Usage: $0 data_dir KG L iters" << endl;
    exit(0);
  }

  anns::SIFTDataset dataset(argv[1]);
  int KG = atoi(argv[2]);
  unsigned L = atoi(argv[3]);
  unsigned iters = atoi(argv[4]);
  dataset.LoadBase(base_data, nb, d);
  dataset.LoadQuery(query_data, nq, d);
  dataset.LoadGroundtruth(gt_data, nq, topk);

  if (KG > BLOCK_DIM_X || d > BLOCK_DIM_X) {
    cout << "KG/d must be samller or equal than BLOCK_DIM_X"
         << " (" << BLOCK_DIM_X << ")" << endl;
    exit(0);
  }

  knng::NNDescent index(base_data, nb, KG, d);

  auto t0 = steady_clock::now();
  index.Build(iters);
  auto t1 = steady_clock::now();
  auto duration = duration_cast<seconds>(t1 - t0).count();
  cout << "Build time: " << duration << " s" << endl;

  std::vector<std::vector<unsigned>> result;
  result.resize(nq);

  t0 = steady_clock::now();
  for (size_t i = 0; i < nq; i++) {
    result[i].resize(topk);
    index.Search(query_data + i * d, topk, L, result[i].data());
  }
  t1 = steady_clock::now();
  duration = duration_cast<milliseconds>(t1 - t0).count();
  double ms_per_query = 1.0 * duration / nq;

  double recall = ComputeRecall(nq, topk, gt_data, result);

  cout << "Recall@100: " << recall << endl;
  cout << "Search time: " << ms_per_query << " ms/query" << endl;
  cout << "QPS: " << 1000 / ms_per_query << endl;

  delete[] base_data;
  delete[] query_data;
  delete[] gt_data;

  return 0;
}
