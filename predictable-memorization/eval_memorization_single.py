import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils", "gpt-neox"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from mmap_dataset import MMapIndexedDataset
import logging
import time
import datetime
import torch
import copy
import boto3
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import GPTNeoXForCausalLM
import transformers.utils as transformer_utils
import multiprocessing as mp
import time
from tqdm import trange

def score(model, context_tokens, true_continuation):
    """Calculate memorization score from context tokens and true continuation

    Performs greedy generation from context tokens and calculates memorization score

    Args:
        model (transformers.GPTNeoXForCausalLM): Pythia model instance being evaluated
        context_tokens (torch.Tensor): Context token indicies of shape (batch_size, 32)
        true_continuation (torch.Tensor): True continuation indicies of shape (batch_size, 32)

    Returns:
        accuracies (torch.Tensor): Accuracies of shape (batch_size,)
    """
    with torch.no_grad():
        context_tokens = torch.tensor(context_tokens).to('cuda')
        true_continuation = torch.tensor(true_continuation).to('cuda')

        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 288, min_length = 288)


        accuracies = []
        for i in range(9):
            s1 = slice(32, 32 + 2 ** i)
            s2 = slice(0, 2 ** i)
            accuracies.append((true_continuation[:, s2]  == generations[:,s1]).all(axis=-1))
        all = torch.stack(accuracies, axis=1)
        return all.cpu()

def main():
    # BATCH_SIZE = 1024
    # MODEL = '70m-v0'
    # CHECKPOINT = 1000
    # model = GPTNeoXForCausalLM.from_pretrained(
    #     f"EleutherAI/pythia-{MODEL}",
    #     use_cache=False,
    #     revision = f'step{CHECKPOINT}',
    #     cache_dir=f"/om/user/sunnyd/transformers_cache/"
    # ).half().eval().cuda()
    # dataset = MMapIndexedDataset('/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document', skip_warmup = True)
    # data = dataset[0:BATCH_SIZE]
    # context = data[:, :32].tolist()
    # true_continuation = data[:, 32:288].tolist()
    # all = score(model, context, true_continuation)
    # import pdb; pdb.set_trace()

    import os

    # with open('results/memorization-evals/evals-running/memorization_70m-v0_23000/rank-0.csv', 'rb') as f:
    #     try:  # catch OSError in case of a one line file 
    #         f.seek(-2, os.SEEK_END)
    #         while f.read(1) != b'\n':
    #             f.seek(-2, os.SEEK_CUR)
    #     except OSError:
    #         f.seek(0)
    #     last_line = f.readline().decode()
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
