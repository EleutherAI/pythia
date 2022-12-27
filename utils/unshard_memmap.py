import os
import argparse

import numpy as np
from tqdm import tqdm


def unshard(
    input_file: str,
    num_shards: int,
    output_dir: str,
):
    """Reconstruct a Megatron .bin file from shards""" 
    
    input_dir = os.path.dirname(input_file)
    base_filename = os.path.basename(input_file)[:-19] # remove 00000-of-xxxxx.bin suffix from shard 0's filename

    full_idx_map = None

    # chunk by iterating over file
    print(f"Loading {num_shards} shards from {input_dir}")
    for i in tqdm(range(num_shards)):
        
        shard_filename = os.path.join(input_dir, base_filename) + f"-{i:05}-of-{(num_shards - 1):05}.bin"
        print(shard_filename)
        shard_memmap = np.memmap(shard_filename, mode="r", order="C")

        if not full_idx_map:
            full_idx_map = shard_memmap
        else:    
            np.concatenate([full_idx_map, shard_memmap])

    
    # write full file
    with open(os.path.join(output_dir, base_filename) + ".bin", "wb+") as out_full_file:
        full_idx_map.tofile(out_full_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Shard a single Megatron data .bin file"
    )

    ## CLI args
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to shard 0",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        help="Provide number of shards (The total seen in shard filenames)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Folder to save .bin file into",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    unshard(args.input_file, args.num_shards, args.output_dir)