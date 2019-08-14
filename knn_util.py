'''
python wrapper for knncuda
Author: Dylan (Wenxuan) Wu
Date: 08/2019
'''



import numpy as np 
import knncuda 


def KNN(ref_data, query_data, K):
    '''
    input:
        ref_data: array, N x dim
        query_data: array, M x dim
        K: Scalar
    output:
        dist: M x K
        inds: M x K
    '''

    res = knncuda.knn(np.ascontiguousarray(ref_data.T), np.ascontiguousarray(query_data.T), K)

    return res.knn_dist, res.knn_index








