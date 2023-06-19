import os
import argparse
import numpy as np
import multiprocess as mp

from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None, help='path to wikipedia HFDataset if --dataset wikipedia')    
    parser.add_argument('--output_path', default=None, help='path to wikipedia HFDataset if --dataset wikipedia')
    parser.add_argument('--process', default='range', type=str, help='Detokenize a range or each single step')
    parser.add_argument('--start_idx', default=0, type=int, help='Starding step index')
    parser.add_argument('--end_idx', default=143000, type=int, help='Ending step index')
    parser.add_argument('--nprocs', default=None, type=int, help='number of processes')
    parser.add_argument('--tokenizer', default="EleutherAI/pythia-12b-deduped")
    parser.add_argument('--auth_token', default=None, type=str, help='Huggingface auth token')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_path = args.data_path
    output_path = args.output_path

    def detokenize(file_path):
        raw_lines = np.load(file_path)
        return [{"text": line} for line in tokenizer.batch_decode(raw_lines)]

    assert args.end_idx <= 143000
    assert args.end_idx > args.start_idx

    print(f"processing steps {args.start_idx} to {args.end_idx}")
    file_list = []
    for step_idx in list(range(args.start_idx, args.end_idx)):
        step_file = f"batch{step_idx}_bs1024.npy"
        checkpoint_path = os.path.join(data_path, step_file)

        if args.process == "singular":
            all_lines = detokenize(checkpoint_path)
            json_file_name = os.path.join(output_path, f"detokenized_step_{step_idx}.jsonl")
            print(f"writing to file to {json_file_name}")
            with jsonlines.open(json_file_name, "w") as jsonfile:
                jsonfile.write_all(
                    all_lines
                )
        else:
            file_list.append(checkpoint_path)


    if args.process == "range":
        if args.nprocs == None:
            num_proc = mp.cpu_count()-1
        else:
            num_proc = args.nprocs

        with mp.Pool(processes=num_proc) as pool:
            all_lines = list(
                tqdm(
                    pool.imap_unordered(detokenize, file_list),
                    total=len(file_list)
                    )
                )

        print("writing to file")
        json_file_name = os.path.join(output_path, f"detokenized_step_{args.start_idx}_to_{args.end_idx}.jsonl")
        with jsonlines.open(json_file_name, "w") as jsonfile:
            for lines in tqdm(all_lines):
                jsonfile.write_all(
                    lines
                )