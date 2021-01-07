A toy for building a KNN graph on GPU.

1. Algorithm:

Based on the idea of NNDescent: A point's neighbor's neighbor is likely its neighbor.

a. Initalize a K-degree graph randomly
b. Repeat n times:
      For every point in the graph
          select K nearest points from neighbors and neighbors' neighbors.

2. Setup

$ git clone --recursive https://github.com/KinglittleQ/cuda-knng.git
$ mkdir build && cd build
$ cmake ..
$ make -j

3. Download SIFT1M dataset

See here http://corpus-texmex.irisa.fr/

4. Build a KNN graph and search on it

$ ./test_knng data_dir KG L iters

data_dir: directory of SIFT1M dataset
KG: 'K' in KNN graph, number of links per node
L: maximum length of the search path
iters: number of iterations

5. Performance

Time for building a KNN graph on SIFT1M with K = 128 (running on GPU): 861 s
Searching on the graph we built above with L = 200 (running on CPU with one thread):
  - Speed: 4.233 ms/query
  - Recall@100: 0.987

6. Core code

include/knng/distance.cuh
include/knng/priority_queue.cuh
include/knng/graph.cuh
