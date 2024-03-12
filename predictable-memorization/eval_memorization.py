#!/usr/bin/env python3
# coding=utf-8
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
import pandas as pd

def generate_dataset(batch_size, start_seq_idx, end_seq_idx, 
    using_s3 = False, 
    prefetch_max = 128
):
    """Wrapper function to prefetch pile sequences

    Intended to run in a saperate `multiprocessing.Process`, this function will continuously prefetch
    context tokens and true continuation from s3 and adds them to `mp_queue`

    Args:
        batch_size (int): Batch size of sequences being evaluted
        start_seq_idx (int): Sequence index of first sequence to be evaluated by current rank
        end_seq_idx (int): Sequence index of last sequence to be evalauted by current rank
        mp_queue (multiprocessing.Queue): Instance of multiprocessing Queue, to add sequences into
        using_s3 (bool): If your datasets are located in s3, set this to true
        prefetch_max (int): Maximum number of sequences that can be pre-fetched into the queue
    
    Env Vars:
        MODEL: name of pythia model being evaluated
        SLURM_PROCID: Rank of current process
    """

    # Load Pile dataset
    prefix = '/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document.bin'
    if "deduped" in os.environ['MODEL']:
        prefix = 'orz/pile/deduped/document.bin'
    s3 = boto3.client('s3')
    buff_size = 2049*batch_size*2
    if using_s3 == False:
        mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

    # Iterate over pile and add sequences to mp_queue
    context_tokens = []
    true_continuation = []
    i = 0
    for i in range(start_seq_idx, end_seq_idx + 1, batch_size):
        if using_s3:
            dataset = s3.get_object(
                Bucket = os.environ['BUCKET'], 
                Key = prefix,
                Range = f'bytes={i*2049*2}-{i*2049*2 + buff_size}'
            )
            data = dataset['Body'].read(buff_size)
            data = np.frombuffer(data, dtype = np.uint16).reshape(-1, 2049)
        else:
            data = mmap_ds[i:i+batch_size]
        context_tokens.extend(data[:, :32].tolist())
        true_continuation.extend(data[:,32:288].tolist())
        i += len(context_tokens)

        if len(context_tokens) == batch_size:
            # (start index of batch, context tokens, true continuation)
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

        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 288, min_length = 288)

        accuracies = np.argmin((true_continuation[:, :256] - generations[:, 32:288] == 0).detach().cpu().numpy(), axis=1)
        overlap = np.sum((true_continuation[:, :256] - generations[:, 32:288] == 0).detach().cpu().numpy(), axis=1)
        return accuracies, overlap

def find_missing_blocks(numbers):
    numbers = sorted(numbers)
    missing_blocks = []
    start = numbers[0]
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
    MODEL = os.environ['MODEL']
    CHECKPOINT = int(os.environ['CHECKPOINT'])

    # Distributed initializations
    # os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    # os.environ['MASTER_PORT'] = '12128'
    logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = logging.INFO)
    logging.info(f"Initializing torch distributed with gpus {torch.cuda.device_count()}")

    # Initialize torch distributed
    # torch.cuda.set_device(RANK)
    # torch.cuda.set_device(RANK%2)
    torch.cuda.set_device(0)
    # dist.init_process_group(
    #     "nccl",
    #     world_size = NUM_PROCS,
    #     rank = RANK
    # )
    # store = dist.TCPStore(os.environ['MASTER_ADDR'], port = 12125, 
    #     world_size = NUM_PROCS, is_master = RANK == 0, timeout = datetime.timedelta(hours=3))

    # dist.barrier()
    print("PROCS", NUM_PROCS, "RANK", RANK)

    # Model initialization
    transformer_utils.logging.set_verbosity_error()

    # Calculate start and end sequence indicies
    # total_num_sequences = CHECKPOINT*1024
    total_num_sequences = 1000*1024
    num_sequences_per_proc = total_num_sequences//NUM_PROCS
    base_path = f'results/memorization-dyn-count/evals-running/memorization_{MODEL}_{CHECKPOINT}'
    filename = f'{base_path}/rank-{RANK}.csv'
    # Create directory if it doesn't exist

    # try:  # catch OSError in case of a one line file 
    #     with open(filename, 'rb') as f:
    #         f.seek(-2, os.SEEK_END)
    #         while f.read(1) != b'\n':
    #             f.seek(-2, os.SEEK_CUR)
    #         last_line = f.readline().decode()
    #         start_idx = int(last_line.split(',')[0])
    # except OSError:
    #     start_idx = num_sequences_per_proc*RANK

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

    # Model initialization
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/om/user/sunnyd/transformers_cache/"
    ).half().eval().cuda()
    
    # dist.barrier()
    logging.info("Loaded Model")

    # Run generations
    iters = 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    for start_idx, end_idx in blocks:
        ds = generate_dataset(BATCH_SIZE, start_idx, end_idx)
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
                    accuracies, overlap = score(model, context, true_continuation)

                    for acc, over in zip(accuracies, overlap):
                        # vals = ','.join([str(j) for j in acc.int().tolist()])
                        f.write(f'{idx},{acc},{over}\n')
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
