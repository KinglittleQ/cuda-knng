Build a KNN graph on GPU.

Algorithm:

Based on the idea of NNDescent: A point's neighbor's neighbor is likely its neighbor.
For every point in the graph, select K nearest points from neighbors and neighbors' neighbors.

1. Setup

$ git clone --recursive https://github.com/KinglittleQ/knng.git
$ mkdir build && cd build
$ cmake ..
$ make -j

2. Download SIFT1M dataset

See here http://corpus-texmex.irisa.fr/

3. Build a KNN graph and search on it

$ ./test_knng data_dir KG L

data_dir: directory of SIFT1M dataset
KG: 'K' in KNN graph, number of links per node
L: maximum length of the search path

4. Performance

Time for building a KNN graph on SIFT1M with K = 128: 861 s
Time for answering one query on the graph we built above with L = 200 (running on CPU): 4.233 ms

5. Core code

include/knng/distance.cuh
include/knng/priority_queue.cuh
include/knng/graph.cuh
