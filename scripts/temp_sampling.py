import os, sys
import click

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

WORKER_ID = int(os.environ['WORKER_ID'])

@click.command()
@click.option('--input_dir', type=str, default="../results/deduped")
@click.option('--indices', type=str, default="../results/deduped/indices_complex_0_1.npy")
@click.option('--output', type=str, default="../results/deduped")
@click.option('--num_perturb', type=int, default=200)
@click.option('--model_id', type=str, default='1b-deduped')
@click.option('--checkpoint', type=int, default=10000)
def main(input_dir, indices, output, num_perturb, model_id, checkpoint):
    os.makedirs(f'{output}/sample_2_0/', exist_ok=True)
    output_fname = f'{output}/sample_2_0/{checkpoint}_{WORKER_ID}'
    indices = np.load(indices)

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
        f"EleutherAI/pythia-{model_id}",
        use_cache=False,
        revision = f'step{checkpoint}',
        cache_dir=f"/om/user/sunnyd/transformers_cache/"
    ).half().eval().cuda()

    batch_indices = np.split(indices, np.arange(0, len(indices), len(indices)//16 + 1)[1:])[WORKER_ID]
    data = np.array([dataset[idx.astype(np.int32).item()] for idx in batch_indices])
    context_tokens = torch.tensor(data[:, :32].astype(np.int32)).to('cuda')
    with torch.no_grad():
        gen_orig = model.generate(context_tokens.cuda(), temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)


    gens = []
    for i in tqdm(range(num_perturb)):
        with torch.no_grad():
            gen = model.generate(context_tokens.cuda(), do_sample=True, temperature = 2.0, max_length = 96, min_length = 96)
            gens.append(gen.detach().cpu().numpy())

    results = np.stack(gens)

    np.savez(output_fname, results = results, gen_orig = gen_orig.detach().cpu().numpy(), data = data, indices = batch_indices)

if __name__ == "__main__":
    main()