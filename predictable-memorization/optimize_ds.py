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
@dask.delayed
def load_chunk(path, ptr, total_size, dtype):
    bin_buffer_mmap = np.memmap(path, mode="r", order="C")
    bin_buffer = memoryview(bin_buffer_mmap)
    data = np.frombuffer(bin_buffer, 
                         dtype=dtype, 
                         count=total_size, 
                         offset=ptr).reshape(-1, 2049)
    return data
    

def mmap_dask_array(blocksize=1000, offset=0, max=50000):
    load = dask.delayed(load_chunk)
    chunks = []
    max_idx = min(max, len(dataset))
    for index in tqdm(range(offset, max_idx, blocksize)):
        chunk_size = min(blocksize, max_idx - index)
        path = '/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document.bin'
        ptr = dataset._index._pointers[index]
        dtype = dataset._index.dtype
        count = np.sum(dataset._index._sizes[index:index+chunk_size])
        # Truncate the last chunk if necessary
        chunk = dask.array.from_delayed(
            load(path, ptr, count, dtype),
            shape=(chunk_size, 2049),
            dtype=dataset[0].dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)

    
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
    

def main():
    cluster = SLURMCluster(cores=8,
                        processes=4,
                        memory="32GB",
                        walltime="48:00:00",
                        # project="fiete",
                        queue="normal",
                        job_extra_directives=["--output=logs/%j.out", "--error=logs/%j.out"]
                        )
    cluster.scale(jobs=64)
    print("Dashboard: ", cluster.dashboard_link)

    client = Client(cluster)


    total_size = 10000 * 1024
    job_size = 50000
    #for j in range(total_size // job_size):
    for i in tqdm(range(total_size // job_size)): 
        res_path = f"/om/tmp/memorization/matches-count-a2a-opt/{i}"
        if os.path.exists(res_path):
            print("skipping "+ res_path)
            continue
        x1 = mmap_dask_array(20000, 0, 1000 * 1024)
        x2 = mmap_dask_array(1000, i * job_size, (i+1) * job_size)
        da.to_npy_stack(
            res_path,
    	    da.blockwise(match, 'ijab', x1[0::20, 32:288],
                'ia', x2, 'jb', dtype=int, adjust_chunks={'a': 1, 'b': 1}).squeeze(), 
                axis=1)

if __name__ == "__main__":
    main()
