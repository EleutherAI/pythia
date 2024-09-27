import functools
import dask.array as da
import click
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from utils.mmap_dataset import MMapIndexedDataset

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import numpy as np

import zlib
def zcomplexity(tokenizer, x):
    orig = tokenizer.decode(x).encode('utf-8')
    enc = zlib.compress(orig)
    return len(enc) / len(orig) 

@click.command()
@click.option('--indices', type=str, default="../results/mem_once/indices.npy")
@click.option('--output', type=str, default="../results/mem_once/zcomplexity.csv")
def main(indices, output):
    cluster = SLURMCluster(cores=8,
                        processes=4,
                        memory="32GB",
                        walltime="48:00:00",
                        # project="fiete",
                        queue="normal",
                        job_extra_directives=["--output=../logs/%j.out", "--error=../logs/%j.out"]
                        )
    cluster.scale(jobs=16)
    print("Dashboard: ", cluster.dashboard_link)
    client = Client(cluster)
    dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
    indices = np.load(indices)
    x1 = da.from_array(np.array([dataset[idx.astype(np.int32).item()] for idx in indices]))
    ex = x1[:, 32:288].compute()

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="/om/user/sunnyd/transformers_cache",
    )

    zcomp_n = np.vectorize(functools.partial(zcomplexity, tokenizer), signature='(n)->()')
    zcomp_complex = zcomp_n(ex)

    np.save(output, zcomp_complex)
    pd.DataFrame({"index": indices, "complexity": zcomp_complex}).set_index('index').to_csv(output)

if __name__ == '__main__':
    main()