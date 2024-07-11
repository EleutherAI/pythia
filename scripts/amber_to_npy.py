import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

for i in tqdm(range(100, 120)):
    if os.path.exists(f"/om/tmp/amber_data/{i:03}.npy"):
        continue
    dataset = load_dataset(
        "LLM360/AmberDatasets",
        data_files=f"train/train_{i:03}.jsonl",
        cache_dir="/om/tmp/amber",
        split=None,
    )
    tot_rows = dataset["train"].num_rows
    arr = np.memmap(f"/om/tmp/amber_data/{i:03}.npy", shape=(tot_rows, 2049),
                    mode='w+', dtype=np.int64)
    for j in tqdm(np.arange(0, tot_rows, 10000)):
        arr[j:j+10000] = np.array(dataset["train"][j:j+10000]["token_ids"])
    arr[j:tot_rows] = np.array(dataset["train"][j:tot_rows]["token_ids"])
    arr.flush()
