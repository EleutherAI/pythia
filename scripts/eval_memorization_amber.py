#!/usr/bin/env python3
# coding=utf-8
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils", "gpt-neox"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
import logging
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import transformers.utils as transformer_utils
import multiprocessing as mp
import time
from tqdm import trange
import pandas as pd
from numba import guvectorize


def generate_dataset(dataset, batch_size, start_seq_idx, end_seq_idx, 
    using_s3 = False, 
    prefetch_max = 128
):

    context_tokens = []
    true_continuation = []
    for i in range(start_seq_idx, end_seq_idx + 1, batch_size):
        data = np.array(dataset["train"][i:i+batch_size]["token_ids"])
        context_tokens.extend(data[:, :32].tolist())
        true_continuation.extend(data[:,32:288].tolist())
        i += len(context_tokens)

        if len(context_tokens) == batch_size:
            yield (
                i - len(context_tokens), 
                context_tokens, true_continuation)
            context_tokens = []
            true_continuation = []

    if len(context_tokens) > 0:
        yield (i - len(context_tokens) + 1, context_tokens, true_continuation)
        context_tokens = []
        true_continuation = []
    
    yield (None, None, None)
    
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

        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 96, min_length = 96)

        accuracies = np.argmin((true_continuation[:, :64] - generations[:, 32:96] == 0).detach().cpu().numpy(), axis=1)
        overlap = np.sum((true_continuation[:, :64] - generations[:, 32:96] == 0).detach().cpu().numpy(), axis=1)
        dist = levenshtein_distance(true_continuation[:, :64].detach().cpu().numpy(), generations[:, 32:96].detach().cpu().numpy())
        return accuracies, overlap, dist 

@guvectorize(["void(int64[:,:], int64[:,:], int64[:])"],
             "(n,i),(n,j)->(n)")
def levenshtein_distance(a, b, result):
    d = np.zeros((a.shape[0], a.shape[1]+1, 2))
    for i in range(0, a.shape[1]+1):
        d[:, i, 0] = i
    for j in range(1, b.shape[1]+1):
        d[:, 0, j % 2] = j
        for i in range(1, a.shape[1]+1):
            substitution_cost = (a[:, i-1] != b[:, j-1])
            for k in range(a.shape[0]):
                d[k, i, j % 2] = min(
                        (d[k, i-1, j % 2] + 1,
                         d[k, i, (j-1) % 2] + 1,
                         d[k, i-1, (j-1) % 2] + substitution_cost[k]
                        )
                )
    result[:] = d[:, -1, (b.shape[-1]) % 2]


def find_missing_blocks(numbers):
    numbers = sorted(numbers)
    missing_blocks = []
    start = numbers[0]
    i = 0
    for i in range(numbers[0], numbers[-1]):
        if i not in numbers and start is not None:
            missing_blocks.append((start, i - 1))
            start = None
        elif i in numbers and start is None:
            start = i
    missing_blocks.append((start, i - 1))
    return missing_blocks

def main():
    # Extracting environment variables and miscellaneous initializations
    BATCH_SIZE = 128
    LOG_INTERVAL = 100 # Log every nth batch evals

    # Distributed variables
    RANK = int(os.environ['JOB_ID'])
    # NUM_PROCS = int(os.environ['SLURM_NPROCS'])
    NUM_PROCS = int(os.environ['NUM_BLOCKS'])

    #RANK = int(os.environ['RANK'])
    #LOCAL_RANK = RANK
    #NUM_PROCS = int(os.environ['WORLD_SIZE'])

    # Eval configuration variables

    DATA_CHECKPOINT = int(os.environ.get('DATA_CHECKPOINT'))
    CHECKPOINT = int(os.environ['CHECKPOINT'])
    SUFFIX = os.environ.get('SUFFIX', '')

    print("PROCS", NUM_PROCS, "RANK", RANK)

    # Model initialization


    # Calculate start and end sequence indicies
    dataset = load_dataset("LLM360/AmberDatasets", data_files=f"train/train_{DATA_CHECKPOINT:03}.jsonl", split=None)
    total_num_sequences = dataset["train"].num_rows
    num_sequences_per_proc = total_num_sequences//NUM_PROCS
    base_path = f'../results/memorization-dyn-count/evals-running/memorization_amber_{CHECKPOINT}_{DATA_CHECKPOINT}{SUFFIX}'
    filename = f'{base_path}/rank-{RANK}.csv'

    indices = set()
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df = pd.read_csv(filename, index_col=0)
        indices = indices.union(set(df.index.to_list()))

        missing_idx = set(range(num_sequences_per_proc*RANK,  min(num_sequences_per_proc *(RANK+1)-1, total_num_sequences-1))).difference(indices)
        print(len(missing_idx))
        blocks = find_missing_blocks(missing_idx)
    else:
        blocks = [(num_sequences_per_proc*RANK, min(num_sequences_per_proc *(RANK+1)-1, total_num_sequences-1))]
    print("Processing: ", blocks)

    logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = logging.INFO)
    logging.info(f"Initializing torch distributed with gpus {torch.cuda.device_count()}")

    torch.cuda.set_device(0)
    transformer_utils.logging.set_verbosity_error()

    # Model initialization
    model = LlamaForCausalLM.from_pretrained("LLM360/Amber", revision=f"ckpt_{CHECKPOINT}",
        cache_dir=f"/om/tmp/amber_cache/"
    ).half().eval().cuda()

    # dist.barrier()
    logging.info("Loaded Model")

    # Run generations
    iters = 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    for start_idx, end_idx in blocks:
        ds = generate_dataset(dataset, BATCH_SIZE, start_idx, end_idx)
        with open(filename, 'a') as f:
            while(True):
                try:
                    t = time.time()
                    idx, context, true_continuation = next(ds)
                    if idx is None:
                        break

                    idx = idx
                    logging.info(f"Loading data took {time.time() - t:.3}s")
                    t = time.time()
                    accuracies, overlap, dist = score(model, context, true_continuation)

                    for acc, over, dist in zip(accuracies, overlap, dist):
                        # vals = ','.join([str(j) for j in acc.int().tolist()])
                        f.write(f'{idx},{acc},{over},{dist}\n')
                        idx += 1
                    f.flush()
                    logging.info(f"Generation uptil {idx} took {time.time() - t:.3}s")
                    # dist.barrier()
                    iters += 1
                except StopIteration:
                    break
    
    

    
    # Uploading evals to s3
    # s3 = boto3.client('s3')
    # s3.put_object(
    #     Body = '\n'.join(memorization_evals).encode(),
    #     Bucket = os.environ['Bucket'],
    #     Key = f'memorization-evals/evals-running/memorization_{MODEL}_{CHECKPOINT}/rank-{RANK}.csv'
    # )
    # dist.barrier()

    return
if __name__ == '__main__':
    main()
