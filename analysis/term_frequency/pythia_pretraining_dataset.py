import os
import numpy as np
import multiprocess as mp

import datasets

from tqdm import tqdm
from collections import Counter
from transformers import GPTNeoXTokenizerFast

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-v1.1-12b-deduped")

time = "minute|hour|day|week|month|year|decade"
punct = re.escape(string.punctuation)
search_string_unary = re.compile(r' (\d){1,5} ')
search_string_time_unit_a = re.compile(f'(?=( (\d){{1,5}} ((\w|[{punct}])+\s+){{0,3}}?({time})))')
search_string_time_unit_b = re.compile(f'(?=( ({time}) ((\w|[{punct}])+\s+){{0,3}}?(\d){{1,5}}))')
search_string_lazy = re.compile(f'(?=( (\d){{1,5}} ((\w|[{punct}])+\s+){{1,3}}(\d){{1,5}}))')
search_string_greedy = re.compile(f'(?=( (\d){{1,5}} ((\w|[{punct}])+\s+){{1,3}}?(\d){{1,5}}))')

def add_freq(freq_dict, key):
    if key in freq_dict:
        freq_dict[key] += 1
    else:
        freq_dict[key] = 1

    return freq_dict

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value + dict_1[key]
   return dict_3

def count_for_checkpoint(path, checkpoint, output_path):

    def count_from_line(file_path):
        frequency_count = {}

        raw_lines = np.load(file_path)
        lines = tokenizer.batch_decode(raw_lines)

        for line in lines:
            line = " "+line