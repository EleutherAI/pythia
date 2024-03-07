#!/usr/bin/env python3
# coding=utf-8

"""
This file contains the code for Converting Scored Jsonl documents into megatron format

Example Usage: torchrun -nproc-per-node 8 convert_dataset.py
"""

import os
import torch
import argparse
import socket
import copy
import spacy
import time
import shutil
from spacy import Language
from typing import Iterable, Tuple, Any
from load_jsonl import LocalJsonlLoader
from argparse import Namespace
from enum import Enum
from glob import glob
import numpy as np
from tqdm.auto import tqdm
import torch.distributed as dist
from streaming import MDSWriter, StreamingDataset

from transformers import AutoTokenizer, PreTrainedTokenizerBase

class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

# Initialize parallel

def init_distributed(rank: int, world_size: int):
    """Initializes torch distributed group

    Args:
        rank (int): Rank of current process
        world size (int): Total number of processes
    """
    dist.init_process_group(backend = 'gloo', rank = rank, world_size = world_size)


# Training tokenizer
def train_tokenizer(args):
    """Trains a tokenizer based on the arguments provided, saves it and returns the tokenizer

    Args:
        args (Namespace): Input arguments
    
    Returns:
        (PretrainedTokenizerBase) tokenizer with additional sentinel tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f'{tokenizer=} must be of type PreTrainedTokenizerBase')
    
    tokenizer.model_max_length = int(1e30)
    if args.bos_text + args.eos_text == '':
        test_tokens = tokenizer('test')
        if(
            test_tokens['input_ids'][0] != tokenizer.bos_token_id and 
            test_tokens['input_ids'][-1] != tokenizer.eos_token_i
        ):
            tok_error_msg = (
                'This tokenizer does not insert an EOS nor BOS token. '
                'Concatenating with this tokenizer will result in sequences being '
                'attached without a separating token. Please use another tokenizer, '
                'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                '--bos_text=<|endoftext|>.'
            )
            raise ValueError(tok_error_msg)
    
    tokenizer_path = f"./{args.tokenizer.replace('/', '_')}-{args.num_sentinels}-special-tokens"
    if args.rank == 0:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"<|val{x}|>" for x in range(0, args.num_sentinels)]}
        )
        tokenizer.save_pretrained(tokenizer_path)
    if args.world_size != 1:
        dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Sentence labels
def score_to_label(args, tokenizer, score):
    """Returns sentinel token for a sentence based on it's toxicity score

    Args:
        args (Namespace): Input arguments
        tokenizer (PreTrainedTokenizerBase): Tokenizer with additional tokens trained
        score (float): Score of current sentence
    """
    val = 0
    for cutoff in args.sentinel_cutoffs:
        if cutoff < score:
            val+=1
        else:
            return tokenizer.additional_special_tokens_ids[val]
    
    return tokenizer.additional_special_tokens_ids[-1]

def tokenize_sentences(args):
    """Tokenizes sentences and yields tokens of sentences

    Args:
        args (Namespace): Input arguments
    
    Returns:
        (bytes): Tokenized tensor in bytes
    """
    # Initialize dataset
    if os.path.isdir(args.dataset_path):
        args = copy.deepcopy(args)
        data_files = glob(f'{args.dataset_path}/*')
        for file in data_files:
            args.dataset_path = file
            yield from tokenize_sentences(args)
        return

    dataloader = LocalJsonlLoader(args.batch_size, args.world_size, args.rank)
    dataloader.load(os.path.join(args.dataset_path))
    ds_iter = iter(dataloader)

    tokenizer = train_tokenizer(args)
    
    ds_iter = tqdm(
        ds_iter, 
        position=args.rank, 
        desc = f'rank-{args.rank}: Iterating through data',
        total = len(dataloader)
    )

    token_ids_buffer = []
    for batch in tqdm(ds_iter):
        all_sents = []
        all_scores = []
        for document in batch:
            all_sents.extend(document['sentences'])
            all_scores.extend(document['scores'])
        
        all_sent_tokens = tokenizer(all_sents)['input_ids']

        if args.concat_mode == ConcatMode.NO_CONCAT:
            for sent in all_sents:
                yield {'tokens': np.asarray(sent, dtype=np.int64).tobytes()}
            break
        
        
        for tokens, score in zip(all_sent_tokens, all_scores):
            if args.bos_text != '':
                token_ids_buffer.append(tokenizer.bos_token_id)
            if np.random.uniform() < args.label_prob:
                token_ids_buffer.append(score_to_label(args, tokenizer, score))
            token_ids_buffer.extend(tokens)
            if args.eos_text != '':
                token_ids_buffer.append(tokenizer.eos_token_id)
            
            if len(token_ids_buffer) >= args.concat_tokens:
                array = np.asarray(token_ids_buffer[:args.concat_tokens], dtype=np.int64).tobytes()
                yield {'tokens': array}
                token_ids_buffer = token_ids_buffer[args.concat_tokens:] if args.should_wrap else []
    dataloader.close()
    if args.world_size != 1:    
        dist.barrier()
if __name__ == '__main__':
    LABEL_PROB = 0.0
    parser = argparse.ArgumentParser(
        prog = 'Converts sentencized documents into megatron format and saves them',
    )
    parser.add_argument(
        '--dataset_path',
        default = f'/weka/orz/pythia/pile-sentencized/',
        help = 'Path to directory of sentencized jsonl files',
    )
    parser.add_argument(
        '--batch_size',
        default = 512,
        type = int,
        help = 'Batch size while tokenizing sentences',
    )
    parser.add_argument(
        '--concat_tokens',
        default = 2049,
        type = int,
        help = (
            'Convert text to tokens and concatenate up to this many tokens. '
            'Set it to -1 if you do not intend to concatenate sentences'
        ),
    )
    parser.add_argument(
        '--tokenizer',
        default = 'EleutherAI/pythia-70m',
        help = 'Tokenizer to use while tokenizing sentences',
    )
    parser.add_argument(
        '--eos_text',
        default = '<|endoftext|>',
        help = 'End of sequence text to use while tokenizing sentences',
    )
    parser.add_argument(
        '--bos_text',
        default = '',
        help = 'Beginning of sequence text to use while tokenizing sentences',
    )
    parser.add_argument(
        '--num_sentinels',
        default = 2,
        help = 'Number of sentinel tokens to add while tokenizing the dataset',
    )
    parser.add_argument(
        '--sentinel_cutoffs',
        default = [5.6e-4],
        help = 'Sentinel cutoffs to rank toxicity into different buckets',
    )
    parser.add_argument(
        '--save_dir',
        default = f'/weka/orz/pythia/pile-converted/prob-{LABEL_PROB}/',
        help = 'Path to save resultant mds files',
    )
    parser.add_argument(
        '--temp_save_dir',
        default = f'/weka/orz/temp/{LABEL_PROB}/',
        help = (
            'Path to temporarily save data in a rank. '
            'These are then combined to be saved in `save_dir`'
        ),
    )
    parser.add_argument(
        '--label_prob',
        default = LABEL_PROB,
        help = 'probability of using a sentinel token at any given place'
    )
    parser.add_argument(
        '--should_wrap',
        default=True,
        help=(
            'Should the leftover tokens after `concat_tokens` warp to next sequence, '
            'in case where `concat_tokens` > 0'
        ) 
    )
    parser.add_argument(
        '--compression',
        default=None,
        choices=['br', 'bz2', 'gz', 'snappy', 'zstd'],
        help='Compression to use while saving the dataset'
    )
    # Initialize distributed
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.rank, args.world_size = rank, world_size
    if args.world_size != 1:
        init_distributed(rank, world_size)

    if args.rank == 0:
        print("*"*10 + "INPUT CONFIG:" + "*"*10)
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print("Hostname", socket.gethostname())
        print("*"*28)
    
    if args.concat_tokens <= 0:
        args.concat_mode = ConcatMode.NO_CONCAT
    else:
        args.concat_mode = ConcatMode.CONCAT_TOKENS
    
    columns = {'tokens': 'bytes'}
    save_dir_temp = os.path.join(args.temp_save_dir, str(args.rank))
    with MDSWriter(columns=columns, out=save_dir_temp) as out:
        for sent_data in tokenize_sentences(args):
            out.write(sent_data)
    
    if args.world_size != 1: dist.barrier()
    if args.rank == 0:
        with MDSWriter(
            columns=columns,
            out=args.save_dir,
            compression=args.compression,
        ) as combined:
            for rank in range(args.world_size):
                rank_save_dir_temp = os.path.join(args.temp_save_dir, str(args.rank))
                reader = StreamingDataset(
                    local=rank_save_dir_temp, 
                    shuffle=False,
                )
                for document in tqdm(reader, desc=f'Combining from rank: {rank}'):
                    combined.write(document)
    
    
    if args.world_size != 1:    
        dist.barrier()
    