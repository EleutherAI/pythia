import click
import pandas as pd
import os
import polars as pl
import dask.array as da
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import numpy as np


@click.command()
@click.option('--input_dir', type=str, default="/om/tmp/memorization/mem_once")
@click.option('--indices', type=str, default="../results/mem_once/indices.npy")
@click.option('--output', type=str, default="../results/mem_once")
def main(input_dir, indices, output):
    cluster = SLURMCluster(cores=8,
                    processes=4,
                    memory="32GB",
                    walltime="48:00:00",
                    # project="fiete",
                    queue="normal",
                    job_extra_directives=["--output=../logs/%j.out", "--error=../logs/%j.out"]
                    )
    cluster.scale(jobs=32)
    print("Dashboard: ", cluster.dashboard_link)
    indices = np.load(indices)

    client = Client(cluster)
    matches = []
    dirs = [int(d) for d in os.listdir(input_dir)]
    for i in sorted(dirs):
        matches.append(da.from_npy_stack(f'{input_dir}/{i}/'))
    m = da.concatenate(matches, axis=1)

    # counts = []
    # bsize = 1000 * 1024
    # topk_range = list(range(0, 9))
    # for k in topk_range:
    #     counts.append(da.topk(m[:, k*bsize:(k+1)*bsize], 10, axis=1))
    # m_topk = da.stack(counts).compute()
    # checkpoint, index, kth = np.unravel_index(np.arange(len(m_topk.reshape(-1))), m_topk.shape)
    # pd.DataFrame({"count": m_topk.reshape(-1), "checkpoint": checkpoint*1000 + 10000, "index": indices[index], "kth": kth}).to_csv(f"{output}/topk.csv")

    bsize = 1000 * 1024
    repeat_count_range = list(range(1, 65, 4))
    repeat_counts = []
    for k in range(10):
        counts = []
        for i in repeat_count_range:
            counts.append(da.sum(m[:, k*bsize:(k+1)*bsize] >= i, axis=1))
        repeat_counts.append(da.stack(counts, axis=1))
    
        # np.save(f"repeat_count_{k}-01.npy", da.stack(counts, axis=1).compute())
    repeat_count = da.stack(repeat_counts, axis=0).compute()
    checkpoint, index, size = np.unravel_index(np.arange(len(repeat_count.reshape(-1))), repeat_count.shape)  
    pd.DataFrame({ "count": repeat_count.reshape(-1), "checkpoint": checkpoint*1000 + 10000, "index": indices[index], "size": np.array(repeat_count_range)[size]}).to_csv(f"{output}/repeat_count.csv")


if __name__ == "__main__":
    main()