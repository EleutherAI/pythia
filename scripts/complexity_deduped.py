import zlib
import numpy as np
import functools
from utils.mmap_dataset import MMapIndexedDataset
from utils.dask import mmap_dask_array
from transformers import GPTNeoXForCausalLM, AutoTokenizer

dataset = MMapIndexedDataset('/om/user/sunnyd/document.bin', skip_warmup=True)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="/om/user/sunnyd/transformers_cache",
)
# x1 = mmap_dask_array(dataset, 20000, 10000 * 1024, 11000 * 1024)
x1 = mmap_dask_array(dataset, 20000, 19000 * 1024, 20000 * 1024)
ex = x1[:, 32:96].compute()

def zcomplexity(tokenizer, x):
  orig = tokenizer.decode(x).encode('utf-8')
  enc = zlib.compress(orig)
  return len(enc) / len(orig) 
zcomp_n = np.vectorize(functools.partial(zcomplexity, tokenizer), signature='(n)->()')
zcomp_complex = zcomp_n(ex)
np.save('../results/zcomplexity_19k', zcomp_complex)