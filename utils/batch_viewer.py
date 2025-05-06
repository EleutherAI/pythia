from mmap_dataset import MMapIndexedDataset
from tqdm import trange
import numpy as np
import argparse
import os

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
        "load_path",
        type = str,
        default = '/mnt/ssd-1/pile_preshuffled/standard/document',
        help = ("MMap dataset path with .bin and .idx files. Omit the .bin (or) .idx "
                "Extension while specifying the path")
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="token_indicies",
        help="Save path for files"
    )
    args = parser.parse_known_args()[0]
    os.makedirs(args.save_path, exist_ok=True)
    filename = os.path.join(args.save_path, "indicies.npy")

    dataset = MMapIndexedDataset(args.load_path, skip_warmup = True)
    indicies = dataset[args.start_iteration*1024: (args.end_iteration+1)*1024]
    np.save(filename, indicies)

