import os, sys
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
import polars as pl
import dask.array as da
import torch.nn.functional as F
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

MODEL='1b-v0'
CHECKPOINT=10000
idx_dir = os.environ.get("IDX_DIR", "../results/indices/0_0.npy")
OUTPUT=os.environ.get("OUTPUT")
# indices = np.load(idx_dir)
indices = np.arange(21000 * 1024 + 1000, 21000 * 1024 + 5000)
dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision=f"step{CHECKPOINT}",
  cache_dir="/om/user/sunnyd/transformers_cache",
)

model_orig = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{MODEL}",
    use_cache=False,
    revision = f'step{CHECKPOINT}',
    cache_dir=f"/om/user/sunnyd/transformers_cache/"
).cuda()


    
def compute_dist(batch):
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/om/user/sunnyd/transformers_cache/"
    ).cuda()
    
    batch = torch.tensor(batch.astype(np.int32)).type(torch.LongTensor).cuda()
    x, y = batch[:, :-1], batch[:, 1:]
    opt = torch.optim.SGD(
                params=model.parameters(),
                lr=1e-5,
                # weight_decay=1e-1,
                # betas=(0.9, 0.95),
                # optim_bits=8,
                # fused=True,
            )
    with torch.autocast(device_type="cuda", enabled=True):
        for i in range(2):
            opt.zero_grad()
            z = model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
            if i == 0:
                starting_loss = loss.detach().cpu().numpy()
            # bar.set_description(f'loss: {loss.detach().cpu()}')
            loss.backward()
            if loss < 0.5:
                break
            opt.step()
            
    param1 = dict(model.named_parameters())
    total_dist = 0
    for k, p2 in model_orig.named_parameters():
        total_dist += torch.sum(torch.square(p2 - param1[k]))
    return i, loss.detach().cpu().numpy(), total_dist.detach().cpu().numpy()

dists = []
for idx in tqdm(indices):
    batch = dataset[idx.astype(np.int32).item()][None, :96]
    dists.append(compute_dist(batch))

# Mkdir if not exists
steps, final_loss, total_dist = zip(*dists)
steps = np.array(steps)
final_loss = np.array(final_loss)
total_dist = np.array(total_dist)
os.makedirs('../results/grad_steps', exist_ok=True)
np.savez(f'../results/grad_steps/{OUTPUT}', steps=steps, final_loss=final_loss, total_dist=total_dist)