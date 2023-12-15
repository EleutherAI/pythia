#!/usr/bin/env python3
# coding=utf-8

"""
This file contains the code for Scoring JsonL documents and saving them

Example Usage: torchrun -nproc-per-node 8 score_detoxify.py
"""

import torch
from typing import Iterable, Tuple, Any
from load_jsonl import LocalJsonlLoader
import numpy as np
from tqdm.auto import tqdm
from detoxify import Detoxify
import torch.distributed as dist
import argparse
import socket
import os
import spacy

def init_distributed(rank: int, world_size: int):
    """Initializes torch distributed group

    Args:
        rank (int): Rank of current process
        world size (int): Total number of processes
    """
    dist.init_process_group(backend = "nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)

def get_raw_text_and_meta(documents: Iterable[dict[str, Any]]) -> Iterable[Tuple[str]]:
    for document in documents:
        yield document['text'], None

def split_sentences(
    documents: Iterable[dict[str, Any]],
    spacy_model,
) -> Iterable[dict[str, Any]]:
    raw_texts = get_raw_text_and_meta(documents)
    for idx, (spacy_doc, meta) in enumerate(spacy_model.pipe(raw_texts, n_process=4, as_tuples=True)):
        for sent in spacy_doc.sents:
            yield {
                'sentence': sent.text_with_ws,  
                'idx': idx,
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = '',
    )
    parser.add_argument(
        '--dataset_path',
        default = '/fsx/orz/temp/val.jsonl',
        help = 'Path to dataset of jsonl file'
    )
    parser.add_argument(
        '--batch_size',
        default = 128,
        type = int,
        help = 'Batch size while classifying sentences'
    )
    parser.add_argument(
        '--classifier_batch_size',
        default = 200,
        type = int,
        help = 'Batch size of Detoxify classifier'
    )
    parser.add_argument(
        '--save_dir',
        default = '/fsx/orz/temp/',
        help = 'Path to save resultant jsonl directory'
    )

    # Initialize distributed
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    args.rank, args.world_size = rank, world_size
    init_distributed(rank, world_size)

    if args.rank == 0:
        print("*"*10 + "INPUT CONFIG:" + "*"*10)
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print("Hostname", socket.gethostname())
        print("*"*28)

    # Initialize dataset
    dataloader = LocalJsonlLoader(args.batch_size, args.world_size, args.rank)
    dataloader.load(args.dataset_path, args.save_dir)
    ds_iter = iter(dataloader)

    # Initialize models
    spacy_model = spacy.blank("en")
    sentencizer = spacy_model.add_pipe("sentencizer")
    spacy_model.max_length = 1e12
    detoxify_model = Detoxify('original', device=f'cuda:{torch.cuda.current_device()}')
    
    ds_iter = tqdm(ds_iter, position=args.rank, desc = f'rank-{args.rank}: Iterating through data')
    for idx, batch in enumerate(ds_iter):
        results = [{'sentences': []} for i in range(args.batch_size)]
        for sent in split_sentences(batch, spacy_model):
            results[sent['idx']]['sentences'].append(sent['sentence'])

        for idx, document in enumerate(batch):
            results[idx]['text'] = document['text']
            results[idx]['meta'] = document['meta']
            sentences = results[idx]['sentences']
            
            scores = []
            # Some documents have unpredictable number of sentences, which causes OOM
            for i in range(0, len(sentences), args.classifier_batch_size):
                batch = sentences[i:i+args.classifier_batch_size]
                scores.extend(detoxify_model.predict(batch)['toxicity'])

            results[idx]['scores'] = scores
            if len(scores) == 0:
                raise ValueError(f"Sequence {results[idx]['text']} has no sentences {results[idx]['sentences']}")
            results[idx]['new_meta'] = {
                'avg_score': np.mean(scores),
                'num_sents': len(sentences)
            }
        dataloader.save(results)
        idx += 1        
    
    dist.barrier()