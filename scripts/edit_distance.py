import numpy as np

# add parent directory to path
import os, sys
sys.path.append('..')

from utils.mmap_dataset import MMapIndexedDataset
import dask
import dask.array as da
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from dask.distributed import Lock
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from numpy.lib.stride_tricks import sliding_window_view
from dask.distributed import Client
from numba import guvectorize

from dask_jobqueue import SLURMCluster

dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
    
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
    
@guvectorize(["void(int64[:,:], int64[:,:], int64[:,:])"],
             "(n,i),(m,j)->(n,m)")
def levenshtein_distance(a, b, result):
    d = np.zeros((a.shape[0], b.shape[0], a.shape[1]+1, 2))
    for i in range(0, a.shape[1]+1):
        d[:, :, i, 0] = i
    for j in range(1, b.shape[1]+1):
        d[:, :, 0, j % 2] = j
        for i in range(1, a.shape[1]+1):
            substitution_cost = (a[:, None, i-1] != b[:, j-1])
            for k in range(a.shape[0]):
                for l in range(b.shape[0]):
                    d[k, l, i, j % 2] = min(
                            (d[k, l, i-1, j % 2] + 1,
                             d[k, l, i, (j-1) % 2] + 1,
                             d[k, l, i-1, (j-1) % 2] + substitution_cost[k, l]
                            )
                    )
    result[:] = d[:, :, -1, (b.shape[-1]) % 2]

def lev_all(a, b):
    return np.expand_dims(np.expand_dims(levenshtein_distance(a, b), -1), -1)

def main():
    cluster = SLURMCluster(cores=8,
                        processes=4,
                        memory="32GB",
                        walltime="12:00:00",
                        # project="fiete",
                        queue="normal",
                        job_extra_directives=["--output=logs/%j.out", "--error=logs/%j.out"]
                        )
    cluster.scale(jobs=64)
    print("Dashboard: ", cluster.dashboard_link)
    indices = np.load('0_1.npz.npy')
    
    x1 = da.from_array(np.array([dataset[idx.astype(np.int32).item()] for idx in indices]))

    client = Client(cluster)


    total_size = 10000 * 1024
    job_size = 10000
    #for j in range(total_size // job_size):
    base_path = "/om/tmp/memorization/matches-count-a2a-opt-10k-lev-01/"
    os.makedirs(base_path, exist_ok=True)

    for i in tqdm(range(total_size // job_size)): 
        res_path = f"{base_path}{i}"
        if os.path.exists(res_path):
            print("skipping "+ res_path)
            continue
        # x1 = mmap_dask_array(20000, 0, 11000 * 1024)
        x2 = mmap_dask_array(50, 10000 * 1024 + i * job_size, 10000 * 1024 + (i+1) * job_size)
        da.to_npy_stack(
            res_path,
    	    da.blockwise(lev_all, 'ijab', x1[:, 32:96],
                'ia', x2, 'jb', dtype=int, adjust_chunks={'a': 1, 'b': 1}).squeeze(), 
                axis=1)

if __name__ == "__main__":
    main()
