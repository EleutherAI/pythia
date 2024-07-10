import os
import pandas as pd

num_blocks = {i:16 for i in range(10000, 30001, 1000)}
# num_blocks[7000] = 8
# num_blocks[8000] = 8

indices = set()
num_remaining = []
OFFSET = 10000 * 1024
for CHECKPOINT in range(10000, 30001, 1000):
    base_path = f'../results/memorization-dyn-count/evals-running/memorization_1b-v0_{CHECKPOINT}_{OFFSET}_lev'
    total_num_sequences = 1000*1024
    block_size = total_num_sequences // num_blocks[CHECKPOINT]
    for RANK in range(num_blocks[CHECKPOINT]):
        file = f'rank-{RANK}.csv'
        if not os.path.exists(os.path.join(base_path, file)) or os.path.getsize(os.path.join(base_path, file)) <= 0:
            # print(os.path.join(base_path, file), os.path.getsize(os.path.join(base_path, file)))
            print(CHECKPOINT, ' ', RANK, ' missing')
            num_remaining.append((CHECKPOINT, RANK, block_size))
            continue
        df = pd.read_csv(os.path.join(base_path, file), index_col=0)
        indices = set(df.index.to_list())
        missing_idx = set(range(OFFSET + block_size * RANK, OFFSET + min(block_size * (RANK+1), total_num_sequences-1))).difference(indices)
        num_remaining.append((CHECKPOINT, RANK, len(missing_idx)))

pd.DataFrame(num_remaining, columns=['checkpoint', 'rank', 'num_remaining']).to_csv('num_remaining.csv', index=False)
