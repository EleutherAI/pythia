import os
import json

import argparse
from typing import Literal
from tqdm import tqdm

import numpy as np
import pandas as pd

from megatron.data import data_utils
from megatron.neox_arguments import NeoXArgs

from megatron import mpu


def view_data(
    args,
    neox_args,
    batch_fn: callable = None,
    save_path: str = None,
):
    # fake MPU setup (needed to init dataloader without actual GPUs or parallelism)
    mpu.mock_model_parallel()
   
    # overrides to config
    neox_args.update_value("train_micro_batch_size_per_gpu", 1024)
    neox_args.update_value("train_batch_size", 1024)
    neox_args.update_value("num_workers", 8)

    # init dataloader
    train_dataloader, _, _ = data_utils.build_train_valid_test_data_iterators(neox_args=neox_args)
    
    print(f"Starting batch iteration from step {args.start_iteration} until step {args.end_iteration}")
    # iterate over dataloader    
    for i in tqdm(range(args.start_iteration, args.end_iteration)):
        batch = next(train_dataloader)["text"].cpu().numpy()
        
        if args.mode == "save":
            # save full batches for each step in the range (WARNING: this may consume lots of storage!)
            filename = f"batch{i}_bs{neox_args.train_micro_batch_size_per_gpu}"
            np.save(os.path.join(save_path, filename), batch)
        elif args.mode == "custom":
            # dump user_defined statistic to a jsonl file (save_fn must return a dict)
            log = batch_fn(batch, i)

            filename = "stats.jsonl"
            with open(os.path.join(save_path, filename), "w+") as f:
                f.write(json.dumps(log) + "\n")
        else:
            raise ValueError(f'mode={mode} not acceptable--please pass either "save" or "custom" !')

        del batch
    print(f"Finished iteration from step {args.start_iteration} to {args.end_iteration}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="What train step to start logging"
    )
    parser.add_argument(
        "--end_iteration",
        type=int,
        default=143000,
        help="Train step to end logging (inclusive)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["save", "custom"],
        help="Choose mode: 'save' to log all batches, and 'custom' to use user-defined statistic"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=0,
        help="Save path for files"
    )
    args = parser.parse_known_args()[0]

    # init neox args
    neox_args = NeoXArgs.consume_deepy_args()
    # set start iter for dataloader
    neox_args.update_value("iteration", args.start_iteration)


    def save_fn(batch: np.array, iteration: int): 
        # define your own logic here
        return {"iteration": iteration, "text": None}

    os.makedirs(args.save_path, exist_ok=True)

    view_data(
        args,
        neox_args,    
        batch_fn=save_fn,
        save_path=args.save_path,
    )
