'''
python wrapper for knncuda
Author: Dylan (Wenxuan) Wu
Date: 08/2019
'''






import numpy as np 
import knn_util
import time


ref_num = 16384
query_num = 4096
dim = 3
k = 16

ref_data = np.random.rand(ref_num, dim)
query_data = np.random.rand(query_num, dim)

num_iter = 100

start_t = time.time()
for i in range(num_iter):

    # note the input array should be contiguous and dim x num, and the output is num x k
    dist, inds = knn_util.KNN(ref_data, query_data, k)

total_time = time.time() - start_t

print("ref_num : ", ref_num)
print("query_num : ", query_num)
print("dim : ", dim)
print("num of neighbours : ", k)
print("mean time over {} iter : ".format(num_iter), total_time / num_iter)









