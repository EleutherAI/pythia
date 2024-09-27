from utils.mmap_dataset import MMapIndexedDataset

prefix = '/om/user/sunnyd/document.bin'
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)