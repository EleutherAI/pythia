from numba import guvectorize
import numpy as np

@guvectorize(["void(int64[:,:], int64[:,:], int64[:,:])"],
             "(n,i),(m,j)->(n,m)")
def match_fn(a, b, result):
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            maxval = 0
            for k in range(b.shape[1]):
                curval = 0
                for l in range(min(a.shape[1], b.shape[1]-k)):
                    if a[i, l] == b[j, k+l]:
                        curval += 1
                    else:
                        break
                maxval = max(maxval, curval)
            result[i, j] = maxval

def match(a, b):
    return np.expand_dims(np.expand_dims(match_fn(a, b), -1), -1)

@guvectorize(["void(int64[:,:], int64[:,:], int64[:])"],
             "(n,i),(n,j)->(n)")
def levenshtein_distance(a, b, result):
    d = np.zeros((a.shape[0], a.shape[1]+1, 2))
    for i in range(0, a.shape[1]+1):
        d[:, i, 0] = i
    for j in range(1, b.shape[1]+1):
        d[:, 0, j % 2] = j
        for i in range(1, a.shape[1]+1):
            substitution_cost = (a[:, i-1] != b[:, j-1])
            for k in range(a.shape[0]):
                d[k, i, j % 2] = min(
                        (d[k, i-1, j % 2] + 1,
                         d[k, i, (j-1) % 2] + 1,
                         d[k, i-1, (j-1) % 2] + substitution_cost[k]
                        )
                )
    result[:] = d[:, -1, (b.shape[-1]) % 2]