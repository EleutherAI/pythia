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
@click.option('--indices', type=str, default="../results/deduped/indices_complex_0_1.npy")
@click.option('--output', type=str, default="../results/deduped")
@click.option('--num_perturb', type=int, default=200)
@click.option('--model_id', type=str, default='1b-deduped')
@click.option('--checkpoint', type=int, default=10000)
def main(indices, output, num_perturb, model_id, checkpoint):
    os.makedirs(f'{output}/perturb/', exist_ok=True)
    output_fname = f'{output}/perturb/{checkpoint}_{WORKER_ID}'
    indices = np.load(indices)

    dataset = MMapIndexedDataset('/om/user/sunnyd/document.bin', skip_warmup = True)
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{model_id}",
        revision="step3000",
        cache_dir="/om2/user/sunnyd/pythia_cache",
    )

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_id}",
        use_cache=False,
        revision = f'step{checkpoint}',
        cache_dir="/om2/user/sunnyd/pythia_cache"
    ).half().eval().cuda()

    batch_indices = np.split(indices, np.arange(0, len(indices), len(indices)//16)[1:])[WORKER_ID]
    data = np.array([dataset[idx.astype(np.int32).item()] for idx in batch_indices])
    context_tokens = torch.tensor(data[:, :32].astype(np.int32)).to('cuda')
    with torch.no_grad():
        gen_orig = model.generate(context_tokens.cuda(), temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)

    model2 = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_id}",
        use_cache=False,
        revision = f'step{checkpoint}',
        cache_dir="/om2/user/sunnyd/pythia_cache"
    ).half().eval().cuda()

    param1 = dict(model.named_parameters())
    gens = []
    for i in tqdm(range(num_perturb)):
        with torch.no_grad():
            for k, p2 in model2.named_parameters():
                p2.copy_(param1[k] + torch.randn_like(p2) * 2e-3)
        with torch.no_grad():
            gen = model2.generate(context_tokens.cuda(), temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)
            gens.append(gen.detach().cpu().numpy())

    results = np.stack(gens)

    np.savez(output_fname, results = results, gen_orig = gen_orig.detach().cpu().numpy(), data = data, indices = batch_indices)

if __name__ == "__main__":
    main()