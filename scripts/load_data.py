from datasets import load_dataset
dataset = load_dataset("LLM360/AmberDatasets", data_files=f"train/train_100.jsonl", split=None, cache_dir='/om2/user/sunnyd/amber_data')
