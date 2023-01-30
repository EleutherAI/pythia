import os
import re
import string
import itertools

from nltk.util import ngrams

import numpy as np
import multiprocess as mp

from tqdm import tqdm
from collections import Counter
from transformers import GPTNeoXTokenizerFast

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-13b-deduped")

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
            matches = re.finditer(search_string_lazy, line)
            for window in [m.group(1) for m in matches]:

                numbers = sorted([int(i) for i in re.findall('\d{1,5}', window)])

                for num in numbers:
                    frequency_count = add_freq(frequency_count, str(num))

                for (num_0, num_1) in itertools.combinations(numbers, 2):
                    key = "{}-{}".format(num_0, num_1)
                    frequency_count = add_freq(frequency_count, key)

            for re_string in [search_string_time_unit_a, search_string_time_unit_b]:
                matches = re.finditer(re_string, line)
                for window in [m.group(1) for m in matches]:
                    numbers = sorted([int(i) for i in re.findall('\d{1,5}', window)])
                    time_units = re.findall(re.compile(f'{time}'), window)

                    for num in numbers:
                        for unit in time_units:
                            key = "{}-{}".format(num, unit)
                            frequency_count = add_freq(frequency_count, key)
        return Counter(frequency_count)

    def iter_count(file_path_list):

        output_list = []
        for file_path in tqdm(file_path_list):
            output_list.append(count_from_line(file_path))

        return output_list

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    checkpoint_path = os.path.join(path, checkpoint)
    all_files = [os.path.join(checkpoint_path, file) for file in os.listdir(checkpoint_path)]

    with mp.Pool(processes=mp.cpu_count()-1) as pool:

        frequency_counts = list(
            tqdm(
                pool.imap_unordered(count_from_line, all_files),
                total=len(all_files)
                )
            )

    freq = Counter()
    for idx, _dict in enumerate(tqdm(frequency_counts)):
        freq.update(_dict)

    save_path = os.path.join(output_path, f"frequency_count_{checkpoint}")
    np.save(save_path, freq, allow_pickle=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Eval on Num Reasoning')
    parser.add_argument('--checkpoint', type=str, default="all")
    parser.add_argument('--input_path', type=str, default="data/")
    parser.add_argument('--output_path', type=str, default="results/")
    args = parser.parse_args()

    if args.checkpoint == "all":
        for checkpoint in os.listdir(args.input_path):
            print(f"Processing {checkpoint}")
            count_for_checkpoint(args.input_path, checkpoint, args.output_path)
    else:
        count_for_checkpoint(args.input_path, args.checkpoint, args.output_path)
