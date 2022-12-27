import os
import argparse

import numpy as np
from tqdm import tqdm


def shard(
    input_file: str,
    output_dir: str,
):
    """Shard a Megatron .bin file into ~ 4.5 GB chunks"""
    SHARD_SIZE = 5_000_000_000 # bytes ~= 4.5 GB 
    
    # load in memmapped .bin file
    full_idx_map = np.memmap(input_file, mode="r", order="C")

    # get number of chunks (rounds down bc start counting from shard number 0)
    num_chunks = full_idx_map.shape[0] // SHARD_SIZE

    # chunk by iterating over file
    for i in tqdm(range(num_chunks + 1)): # while still have file contents remaining to chunk:
        
        start_idx = i * SHARD_SIZE
        end_idx = (i + 1) * SHARD_SIZE

        if end_idx > full_idx_map.shape[0]:
            chunk = full_idx_map[start_idx:]
        else:
            chunk = full_idx_map[start_idx:end_idx]

        shard_filename = os.path.join(output_dir, os.path.basename(input_file)[:-4]) + f"-{i:05}-of-{num_chunks:05}.bin"
        with open(shard_filename, "wb+") as out_shard_file:
            print(f"Dumping shard {i:05} to {shard_filename} ...")
            chunk.tofile(out_shard_file)

        del chunk


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Shard a single Megatron data .bin file"
    )

    ## CLI args
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to .bin file e.g. /path/to/pile_0.87_deduped_text_document.bin",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Folder to save shards into",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shard(args.input_file, args.output_dir)
