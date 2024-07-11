import dask
import numpy as np
from tqdm import tqdm
import dask.array as da

@dask.delayed
def load_chunk(path, ptr, total_size, dtype):
    bin_buffer_mmap = np.memmap(path, mode="r", order="C")
    bin_buffer = memoryview(bin_buffer_mmap)
    data = np.frombuffer(bin_buffer, 
                         dtype=dtype, 
                         count=total_size, 
                         offset=ptr).reshape(-1, 2049)
    return data
    

def mmap_dask_array(dataset, blocksize=1000, offset=0, max=50000):
    load = dask.delayed(load_chunk)
    chunks = []
    max_idx = min(max, len(dataset))
    for index in tqdm(range(offset, max_idx, blocksize)):
        chunk_size = min(blocksize, max_idx - index)
        path = '/om/user/sunnyd/data/datasets--EleutherAI--pile-standard-pythia-preshuffled-merged/document.bin'
        ptr = dataset._index._pointers[index]
        dtype = dataset._index.dtype
        count = np.sum(dataset._index._sizes[index:index+chunk_size])
        # Truncate the last chunk if necessary
        chunk = dask.array.from_delayed(
            load(path, ptr, count, dtype),
            shape=(chunk_size, 2049),
            dtype=dataset[0].dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)