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
@click.option('--output', type=str, default="../results/mem_once")
def main(input_dir, output):
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

    client = Client(cluster)
    matches = []
    dirs = [int(d) for d in os.listdir(input_dir)]
    for i in sorted(dirs):
        matches.append(da.from_npy_stack(f'{input_dir}/{i}/'))
    m = da.concatenate(matches, axis=1)

    counts = []
    bsize = 1000 * 1024
    for k in range(0, 9):
        counts.append(da.topk(m[:, k*bsize:(k+1)*bsize], 10, axis=1))
    m_topk = da.stack(counts).compute()
    np.save(f"{output}/topk.npy", m_topk)

    bsize = 1000 * 1024
    for k in range(10):
        counts = []
        for i in range(1, 65, 4):
            counts.append(da.sum(m[:, k*bsize:(k+1)*bsize] >= i, axis=1))
        # np.save(f"repeat_count_{k}-01.npy", da.stack(counts, axis=1).compute())

    repeat_count = da.stack(counts).compute()
    np.save(f"{output}/repeat_count.npy", repeat_count)

if __name__ == "__main__":
    main()