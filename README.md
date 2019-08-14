# K-Nearest Neighbor using GPU

This repository contains a python wrapper for the [K Nearest Neighbor CUDA library](https://github.com/vincentfpgarcia/kNN-CUDA). The wrapper is written using Pybind11. 


# Installation

Please modify the path for `nvcc`, `cuda`, and install pybind11 in `compile.sh` to make sure it can compile correctly.
The program is tested under python 3.5.

```
./compile.sh
```


# Example

Once you build the wrapper, run

```
python example.py
ref_num :  16384
query_num :  40960
dim :  3
num of neighbours :  16
mean time over 100 iter :  0.2607978343963623
```

# Usage

In python, after you `import knn_util`, you can access the knn function.

## dist, inds = knn_util.KNN(ref_data, query_data, k)

Both `ref_data` and `query_data` should be 2 dimensional numpy arraies with shape `(N x dim)` and `M x dim`. The function will return two outputs: `dist` with shape `(M x K)`  and `inds` with shape `(M x K)`


