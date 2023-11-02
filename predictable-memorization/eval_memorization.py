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

def generate_dataset(batch_size, start_seq_idx, end_seq_idx, mp_queue, 
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
    prefix = '/scratch/pile/standard/document.bin'
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
        true_continuation.extend(data[:,32:64].tolist())
        i += len(context_tokens)

        if len(context_tokens) == batch_size:
            # (start index of batch, context tokens, true continuation)
            mp_queue.put((
                i - len(context_tokens), 
                context_tokens, true_continuation))
            context_tokens = []
            true_continuation = []
            while mp_queue.qsize() > prefetch_max:
                time.sleep(0.05)

    if len(context_tokens) > 0:
        mp_queue.put((i - len(context_tokens) + 1, context_tokens, true_continuation))
        context_tokens = []
        true_continuation = []
    
    mp_queue.put((None, None, None))
    
    



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

        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 64, min_length = 64)


        accuracies = (true_continuation == generations[:,32:64]).float().mean(axis=-1)
        return accuracies.cpu()

def main():
    # Extracting environment variables and miscellaneous initializations
    BATCH_SIZE = 1024
    LOG_INTERVAL = 100 # Log every nth batch evals

    # Distributed variables
    RANK = int(os.environ['SLURM_PROCID'])
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    NUM_PROCS = int(os.environ['SLURM_NPROCS'])

    RANK = int(os.environ['RANK'])
    LOCAL_RANK = RANK
    NUM_PROCS = int(os.environ['WORLD_SIZE'])

    # Eval configuration variables
    MODEL = os.environ['MODEL']
    CHECKPOINT = int(os.environ['CHECKPOINT'])

    # Distributed initializations
    # os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    # os.environ['MASTER_PORT'] = '12128'
    logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = logging.INFO)
    logging.info(f"Initializing torch distributed with gpus {torch.cuda.device_count()}")

    # Initialize torch distributed
    torch.cuda.set_device(RANK)
    dist.init_process_group(
        "nccl",
        world_size = NUM_PROCS,
        rank = RANK
    )
    store = dist.TCPStore(os.environ['MASTER_ADDR'], port = 12125, 
        world_size = NUM_PROCS, is_master = RANK == 0, timeout = datetime.timedelta(hours=3))

    dist.barrier()

    # Model initialization
    transformer_utils.logging.set_verbosity_error()

    # Calculate start and end sequence indicies
    total_num_sequences = CHECKPOINT*1024
    num_sequences_per_proc = total_num_sequences//NUM_PROCS
    start_idx = num_sequences_per_proc*RANK
    end_idx = num_sequences_per_proc*(RANK+1) - 1
    if RANK == (NUM_PROCS -1):
        end_idx = total_num_sequences - 1

    # Dataset Initialization
    mp_queue = mp.Queue()
    ds_process = mp.Process(target = generate_dataset, args=(BATCH_SIZE, start_idx, end_idx, mp_queue))
    ds_process.start()

    # Model initialization
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/fsx/orz/models/"
    ).half().eval().cuda()
    
    dist.barrier()
    logging.info("Loaded Model")

    # Run generations
    memorization_evals = []
    iters = 0
    while(True):
        try:
            t = time.time()
            idx, context, true_continuation = mp_queue.get()
            if idx is None:
                mp_queue.close()
                break

            idx = idx
            logging.info(f"Loading data took {time.time() - t:.3}s")
            t = time.time()
            accuracies = score(model, context, true_continuation)

            for acc in accuracies:
                memorization_evals.append(f'{idx},{acc}')
                idx += 1
            logging.info(f"Generation uptil {idx} took {time.time() - t:.3}s")
            dist.barrier()
            iters += 1
        except StopIteration:
            break
    
    ds_process.join()
    
    # Uploading evals to s3
    s3 = boto3.client('s3')
    s3.put_object(
        Body = '\n'.join(memorization_evals).encode(),
        Bucket = os.environ['Bucket'],
        Key = f'memorization-evals/evals-running/memorization_{MODEL}_{CHECKPOINT}/rank-{RANK}.csv'
    )
    dist.barrier()

    return
if __name__ == '__main__':
    mp.set_start_method('spawn')
    try:
        main()
    except RuntimeError as err:
        import requests
        import datetime
        import socket
        ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')+'UTC'
        resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        print(f'ERROR for {socket.gethostname()} at {ts} on {resp.text} device: {type(err).__name__}: {err}', flush=True)
        raise err