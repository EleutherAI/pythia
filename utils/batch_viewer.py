#!/usr/bin/env python
# coding=utf-8
from megatron import utils as megatron_utils
from megatron.data import data_utils
from megatron.neox_arguments import NeoXArgs
#import result_records
import numpy as np

models = {
    '13B': [143000],
    '6.7B': [143000],
    '2.7B': [143000],
    '1.3B': [71500],
    '800M': [143000],
    '800M_deduped': [143000],
    '350M': [71500],
    '125M': [71500],
}
import os, time
import json
import pandas as pd
from tqdm import tqdm

#megatron_utils.print_rank_0(f"Total Iterations: {total_iters}")

from megatron import mpu

import argparse
from typing import Literal

def view_data(
    args,
    neox_args,
    batch_fn: callable = None,
):

    mpu.mock_model_parallel()
    
    neox_args.update_value("train_micro_batch_size_per_gpu", 1024)
    neox_args.update_value("train_batch_size", 1024)
    neox_args.update_value("num_workers", 8)

 

    # init dataloader
    train_dataloader, _, _ = data_utils.build_train_valid_test_data_iterators(neox_args=neox_args)

    # iterate over dataloader    
    for i in tqdm(range(args.start_iteration, args.end_iteration)):
        batch = next(train_dataloader)["text"].cpu().numpy()
        
        if args.mode == "save":
            # save full batches for each step in the range (WARNING: this may consume lots of storage!)
            np.save(f"./dump_data/batch{i}_bs{neox_args.train_micro_batch_size_per_gpu}", batch)
        elif args.mode == "custom":
            # dump user_defined statistic to a jsonl file (save_fn must return a dict)
            log = save_fn(batch, i)

            with open("./dump_data/stats.jsonl", "w+") as f:
                f.write(json.dumps(log) + "\n")
        else:
            raise ValueError(f'mode={mode} not acceptable--please pass either "save" or "custom" !')

        del batch
    

#import json
#with open(f'memorized-seqs/texts_800M_demo.json', 'w') as f:
#    json.dump(texts, f)

if __name__ == '__main__':
    
    # add args:
    # - MODEL_NAME: selects a config from the Pythia repo
    # - indices to range over
    # - what to log (save each batch in range, save statistic on each batch to a Pandas DF ; each raw batch saved to own batch's file `out_batches/{MODEL_NAME}_batch{IDX}.npy`)
    
    parser = argparse.ArgumentParser(
        description="args",
    )

    parser.add_argument(
        "--start_iteration",
        type=int,
        help="Train step to start at"
    )
    parser.add_argument(
        "--end_iteration",
        type=int,
        default=143000,
        help="Train step to end with (inclusive)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["save", "custom"],
        help="Choose mode: 'save' to log all batches, and 'custom' to use user-defined statistic"
    )

    #TODO:
    # - allow passing of a function to operate on a batch then save output? function defaults to `lambda x: x`
    # set range of iters to iterate over (minimum/start and maximum/stopping)

    args = parser.parse_known_args()[0]

    # init neox args
    neox_args = NeoXArgs.consume_deepy_args()
    # set start iter for dataloader
    neox_args.update_value("iteration", args.start_iteration)

    # settings unique to script
    # neox_args.update_value("train_micro_batch_size_per_gpu", 1024)
    # neox_args.update_value("train_batch_size", 1024)
    # neox_args.update_value("num_workers", 8)
    # print(neox_args.train_micro_batch_size_per_gpu)

    def save_fn(batch: np.array, iteration: int):
        
        # define your own logic here
        return {"iteration": iteration, "text": None}


    view_data(
        args,
        neox_args,    
        batch_fn=save_fn,
    )

