import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
import polars as pl
import dask.array as da
from tqdm import tqdm
from utils.mmap_dataset import MMapIndexedDataset

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import numpy as np

NUM_PERTURB = 200
BATCH_SIZE=256
MODEL='1b-v0'
CHECKPOINT=10000
WORKER_ID = int(os.environ['WORKER_ID'])
SUFFIX = os.environ.get('SUFFIX', '')
INDICES = os.environ.get('INDICES', '')
#all_indices = np.load(INDICES)
output_fname = f'perturbation-{CHECKPOINT}-{WORKER_ID}{SUFFIX}.npz'

def main():
    df = pd.read_parquet('../results/analysis.parquet.gzip')  

    # maxvals = df.groupby(df.index.get_level_values(0)).transform('max') 
    # indices_0_1 = np.array(list(set(df[(df["longest_match"] < 5 ) & (df.index.get_level_values(-1) == 10000)].index.get_level_values(0).to_numpy()).intersection(
    #     set(df[(maxvals["longest_match"] > 10) & (maxvals["cumsum30"] < 10)].index.get_level_values(0).to_numpy()))))
    # indices_0_0 = np.array(list(set(df[(df["longest_match"] < 5 ) & (df.index.get_level_values(-1) == 10000)].index.get_level_values(0).to_numpy()).intersection(
    #     set(df[maxvals["longest_match"] <= 10].index.get_level_values(0).to_numpy()))))
    # indices_0 = np.array(list(set(df[(maxvals["cumsum30"] < 10)].index.get_level_values(0).to_numpy())))[BATCH_SIZE * WORKER_ID: BATCH_SIZE * (WORKER_ID + 1)]
    # indices_1 = np.array(list(set(df[(maxvals["cumsum30"] >= 10)].index.get_level_values(0).to_numpy())))[BATCH_SIZE  * WORKER_ID:  BATCH_SIZE * (WORKER_ID + 1)]

    # indices_0_0 = indices_0_0[np.random.choice(len(indices_0_0), len(indices_0_1))]

    dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="/om/user/sunnyd/transformers_cache",
    )

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/om/user/sunnyd/transformers_cache/"
    ).half().eval().cuda()

    # all_indices = np.concatenate([indices_0, indices_1])

    all_indices = np.arange(21000 * 1024 + 1000 * WORKER_ID, 21000 * 1024 + 1000 * (WORKER_ID+1))
    BATCH_SIZE=len(all_indices) // 16
    # all_indices = all_indices[BATCH_SIZE * WORKER_ID: min(BATCH_SIZE * (WORKER_ID + 1), len(all_indices))]
    # all_indices = np.arange(10000 * 1024 + 500 * WORKER_ID, 10000 * 1024 + 500 * (WORKER_ID+1))
    # all_indices = np.arange(21000 * 1024 + 500 * WORKER_ID, 21000 * 1024 + 500 * (WORKER_ID+1))
    data = np.array([dataset[idx.astype(np.int32).item()] for idx in all_indices])
    context_tokens = torch.tensor(data[:, :32].astype(np.int32)).to('cuda')
    with torch.no_grad():
            gen_orig = model.generate(context_tokens.cuda(), temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)

    model2 = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/om/user/sunnyd/transformers_cache/"
    ).half().eval().cuda()

    param1 = dict(model.named_parameters())
    gens = []
    for i in tqdm(range(NUM_PERTURB)):
        with torch.no_grad():
            for k, p2 in model2.named_parameters():
                p2.copy_(param1[k] + torch.randn_like(p2) * 2e-3)
        with torch.no_grad():
            gen = model2.generate(context_tokens.cuda(), temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)
            gens.append(gen.detach().cpu().numpy())

    results = np.stack(gens)

    np.savez(output_fname, results = results, indices = all_indices, gen_orig = gen_orig.detach().cpu().numpy(), data = data)