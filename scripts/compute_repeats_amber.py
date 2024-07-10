import numpy as np

# add parent directory to path
import os, sys
sys.path.append('..')

import dask
import dask.array as da
from tqdm import tqdm
from dask.distributed import Client
from numba import guvectorize

from dask_jobqueue import SLURMCluster

@dask.delayed
def load_chunk(ckpt_num, ptr, total_size):
    arr = np.memmap(f"/om/tmp/amber_data/{ckpt_num:03}.npy", mode="r", dtype=np.int64).reshape(-1, 2049)
    return arr[ptr:ptr+total_size]
    

def mmap_dask_array(blocksize=1000, offset=0, ckpt_num=100):
    arr = np.memmap(f"/om/tmp/amber_data/{ckpt_num:03}.npy", mode="r", dtype=np.int64).reshape(-1, 2049)
    data_len = arr.shape[0]

    load = dask.delayed(load_chunk)
    chunks = []
    for index in tqdm(range(offset, data_len, blocksize)):
        chunk_size = min(blocksize,  data_len- index)
        # Truncate the last chunk if necessary
        chunk = dask.array.from_delayed(
            load(ckpt_num, index, blocksize),
            shape=(chunk_size, 2049),
            dtype=np.int64
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
                        walltime="12:00:00",
                        # project="fiete",
                        queue="normal",
                        job_extra_directives=["--output=../logs/%j.out", "--error=../logs/%j.out"]
                        )
    cluster.scale(jobs=64)
    print("Dashboard: ", cluster.dashboard_link)
    client = Client(cluster)


    x1 = mmap_dask_array(40000, 0, 100)
    batch_size = 160000
    for i in tqdm(range(100, 110)): 
        x2 = mmap_dask_array(2000, 0, i)
        for j in np.arange(0, x2.shape[0]//batch_size + 1):
            res_path = f"/om/tmp/memorization/matches-count-a2a-opt-amber/{i}-{j}"
            if os.path.exists(res_path):
                print("skipping "+ res_path)
                continue
            os.makedirs(res_path, exist_ok=True)
            da.to_npy_stack(
                res_path,
                da.blockwise(match, 'ijab', x1[::40, 32:96],
                    'ia', x2[j * batch_size: min((j+1) * batch_size, x2.shape[0])], 
                    'jb', dtype=int, adjust_chunks={'a': 1, 'b': 1}).squeeze(), 
                    axis=0)

if __name__ == "__main__":
    main()
