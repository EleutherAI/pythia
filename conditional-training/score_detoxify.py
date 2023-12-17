#!/usr/bin/env python3
# coding=utf-8

"""
This file contains the code for Scoring JsonL documents and saving them

Example Usage: torchrun -nproc-per-node 8 score_detoxify.py
"""

import os
import torch
import argparse
import socket
import spacy
import time
from spacy import Language
from typing import Iterable, Tuple, Any
from load_jsonl import LocalJsonlLoader
from argparse import Namespace
import numpy as np
from tqdm.auto import tqdm
from detoxify import Detoxify
import torch.distributed as dist

# Initialize parallel

def init_distributed(rank: int, world_size: int):
    """Initializes torch distributed group

    Args:
        rank (int): Rank of current process
        world size (int): Total number of processes
    """
    dist.init_process_group(backend = "nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)

# Document parsing functions

def get_raw_text_and_meta(documents: Iterable[dict[str, Any]]) -> Iterable[Tuple[str]]:
    """Yields an iterator that extracts text from jsonl document"""
    for document in documents:
        yield document['text']

def split_sentences(
    documents: Iterable[dict[str, Any]],
    spacy_model: Language,
    args: Namespace
) -> Iterable[dict[str, Any]]:
    """Splits sentences using blank scipy model
    
    Args:
        documents: Jsonl document dictionaries
        spacy_model: Blank En model for splitting sentences
        args: Arguments from argparse.Parser

    Yields:
        Dictionary with sentences from a document and document's corresponding index 
    """
    raw_texts = get_raw_text_and_meta(documents)
    for idx, (spacy_doc) in enumerate(spacy_model.pipe(raw_texts, n_process=os.cpu_count()//args.world_size)):
        all_sentences = []
        for sent in spacy_doc.sents:
            all_sentences.append(sent.text_with_ws)
        yield {
            'sentences': all_sentences,  
            'idx': idx,
        }

def combine_sentences(
    sentences: Iterable[list[str]],
    args: Namespace
) -> Iterable[list[str]]:
    """Combines sentences to make sure every sentence atleast has n thresholded chars
    
    Args:
        sentences: List of sentences
        args: Arguments from argparse.Parser

    Returns:
        Sentences, combined
    """
    res_sents = []
    for sentence in sentences:
        if len(res_sents) == 0:
            res_sents.append(sentence)
        elif (len(res_sents[-1]) < args.sentence_min_char_threshold):
            res_sents[-1] += args.sentence_combine_char + sentence
        else:
            res_sents.append(sentence)
    
    # Last sentence might still be less than thresholded chars
    if (len(sentences[-1]) < args.sentence_min_char_threshold):
        if(len(sentences) > 1):
            sentences[-2] += args.sentence_combine_char + sentences[-1]
            sentences = sentences[:-1]
    
    return res_sents
    
if __name__ == '__main__':
    PARSE_JSONL_FILE = '01'
    parser = argparse.ArgumentParser(
        prog = 'Sentencizes and classifies documents using detoxify from jsonl format',
    )
    parser.add_argument(
        '--dataset_path',
        default = f'/fsx/orz/temp/{PARSE_JSONL_FILE}.jsonl',
        help = 'Path to dataset of jsonl file'
    )
    parser.add_argument(
        '--batch_size',
        default = 1024,
        type = int,
        help = 'Batch size while classifying sentences'
    )
    parser.add_argument(
        '--classifier_batch_size',
        default = 1024,
        type = int,
        help = 'Batch size of Detoxify classifier'
    )
    parser.add_argument(
        '--save_dir',
        default = f'/fsx/orz/temp/{PARSE_JSONL_FILE}/',
        help = 'Path to save resultant jsonl directory'
    )
    parser.add_argument(
        '--sentence_min_char_threshold',
        default = 10,
        help = 'Threshold to combine sentences of less than n chars'
    )
    parser.add_argument(
        '--sentence_combine_char',
        default = '""',
        help = 'Character for combining sentences of less than threshold characters'
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
    detoxify_model.model.half() # manually cast to fp16
    
    ds_iter = tqdm(
        ds_iter, 
        position=args.rank, 
        desc = f'rank-{args.rank}: Iterating through data',
        total = len(dataloader)
    )

    # Iterate and classify
    for idx, batch in enumerate(ds_iter):
        all_sents = []
        all_sent_ids = []
        all_scores = []
        results = [{'sentences': [], 'scores': []} for i in range(len(batch))]

        # Get combined sentences
        for sents in split_sentences(batch, spacy_model, args):
            idx = sents['idx']
            sentences = combine_sentences(sents['sentences'], args)
            all_sents.extend(sentences)
            all_sent_ids.extend([idx for i in range(len(sentences))])

        # Get sentence scores
        for i in range(0, len(all_sents), args.classifier_batch_size):
            sent_batch = all_sents[i:i+args.classifier_batch_size]
            all_scores.extend(detoxify_model.predict(sent_batch)['toxicity'])

        # Store data in results
        for idx, pos in enumerate(all_sent_ids):
            results[pos]['sentences'].append(all_sents[idx])
            results[pos]['scores'].append(all_scores[idx])

        for idx, document in enumerate(batch):
            results[idx]['text'] = document['text']
            results[idx]['meta'] = document['meta']
            scores = results[idx]['scores']

            if len(scores) == 0:
                raise ValueError(f"Sequence {results[idx]['text']} has no sentences {results[idx]['sentences']}")
            results[idx]['new_meta'] = {
                'avg_score': np.mean(scores),
                'num_sents': len(sentences)
            }
        dataloader.save(results)
    
    dataloader.close()    
    dist.barrier()
    dataloader.combine(args.save_dir)
    dist.barrier()