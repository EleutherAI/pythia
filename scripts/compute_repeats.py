import numpy as np

# add parent directory to path
import functools
import click
import os

from utils.mmap_dataset import MMapIndexedDataset
import dask.array as da
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from dask.distributed import Client
from numba import guvectorize
from utils.dask import mmap_dask_array

from dask_jobqueue import SLURMCluster

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
    
@click.command()
@click.option('--indices', type=str, default="../results/mem_once/indices.npy")
@click.option('--output', type=str, default="/om/tmp/memorization/mem_once")
@click.option('--job-size', type=int, default=200000)
@click.option('--offset', type=int, default=10000 * 1024)
@click.option('--total-size', type=int, default=10000 * 1024)
@click.option('--block_size', type=int, default=1000)
def main(indices, output, job_size, offset, total_size, block_size):
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
    dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
    mda = functools.partial(mmap_dask_array, dataset)

    client = Client(cluster)
    indices = np.load(indices)
    x1 = da.from_array(np.array([dataset[idx.astype(np.int32).item()] for idx in indices]))
    os.makedirs(output, exist_ok=True)
    for i in tqdm(range(total_size // job_size)): 
        res_path = f"{output}/{i}"
        if os.path.exists(res_path):
            print("skipping "+ res_path)
            continue
        x2 = mda(block_size, offset + i * job_size, offset + (i+1) * job_size)
        da.to_npy_stack(
            res_path,
    	    da.blockwise(match, 'ijab', x1[:, 32:96],
                'ia', x2, 'jb', dtype=int, adjust_chunks={'a': 1, 'b': 1}).squeeze(), 
                axis=1)

if __name__ == "__main__":
    main()
