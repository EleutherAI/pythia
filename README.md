# [Pythia: Interpreting Autoregressive Transformers Across Time and Scale](https://arxiv.org/pdf/2304.01373.pdf)

This repository is for EleutherAI's project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers. For detailed info on the models, their training, and their behavior, please see [our paper](https://arxiv.org/pdf/2304.01373.pdf).

## Models

| Params              | n_layers | d_model | n_heads | d_head | Batch Size | Learning Rate | Checkpoints                                                | Evaluations     |
| ------------------- | -------- | ------- | ------- | ------ | ---------- | ------------- | ---------------------------------------------------------- | --------------- |
| Pythia-70M          | 6        | 512     | 8       | 64     | 2M         | 1e-3          | [Here](https://huggingface.co/EleutherAI/pythia-70m)          | Ready           |
| Pythia-70M-Deduped  | 6        | 512     | 8       | 64     | 2M         | 1e-3          | [Here](https://huggingface.co/EleutherAI/pythia-70m-deduped)  | Ready           |
| Pythia-160M         | 12       | 768     | 12      | 64     | 2M         | 6e-4          | [Here](https://huggingface.co/EleutherAI/pythia-160m)         | Ready           |
| Pythia-160M-Deduped | 12       | 768     | 12      | 64     | 2M         | 6e-4          | [Here](https://huggingface.co/EleutherAI/pythia-160m-deduped) | Ready           |
| Pythia-410M         | 24       | 1024    | 16      | 64     | 2M         | 3e-4          | [Here](https://huggingface.co/EleutherAI/pythia-410m)         | Ready           |
| Pythia-410M-Deduped | 24       | 1024    | 16      | 64     | 2M         | 3e-4          | [Here](https://huggingface.co/EleutherAI/pythia-410m-deduped) | Ready           |
| Pythia-1B         | 16       | 2048    | 8       | 256   | 2M         | 3e-4          | [Here](https://huggingface.co/EleutherAI/pythia-1b)         | Ready           |
| Pythia-1B-Deduped | 16       | 2048    | 8       | 256    | 2M         | 3e-4          | [Here](https://huggingface.co/EleutherAI/pythia-1b-deduped) | Ready           |
| Pythia-1.4B         | 24       | 2048    | 16      | 128    | 2M         | 2e-4          | [Here](https://huggingface.co/EleutherAI/pythia-1.4b)         | Ready           |
| Pythia-1.4B-Deduped | 24       | 2048    | 16      | 128    | 2M         | 2e-4          | [Here](https://huggingface.co/EleutherAI/pythia-1.4b-deduped) | Ready           |
| Pythia-2.8B         | 32       | 2560    | 32      | 80     | 2M         | 1.6e-4        | [Here](https://huggingface.co/EleutherAI/pythia-2.8b)         | Ready           |
| Pythia-2.8B-Deduped | 32       | 2560    | 32      | 80     | 2M         | 1.6e-4        | [Here](https://huggingface.co/EleutherAI/pythia-2.8b-deduped) | Ready           |
| Pythia-6.9B         | 32       | 4096    | 32      | 128    | 2M         | 1.2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-6.9b)         | Ready           |
| Pythia-6.9B-Deduped | 32       | 4096    | 32      | 128    | 2M         | 1.2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) | Ready           |
| Pythia-12B          | 36       | 5120    | 40      | 128    | 2M         | 1.2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-12b)          | Ready |
| Pythia-12B-Deduped  | 36       | 5120    | 40      | 128    | 2M         | 1.2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-12b-deduped)  | Ready |

We train and release a suite of 8 model sizes on 2 different datasets: [the Pile](https://pile.eleuther.ai/), as well as the Pile with deduplication applied.

All 8 model sizes are trained on the exact same data, in the exact same order. Each model saw 299,892,736,000 ~= 299.9B tokens during training, and *143 checkpoints* for each model are saved every 2,097,152,000 ~= 2B tokens, evenly spaced throughout training. This corresponds to just under 1 epoch on the Pile for non-"deduped" models, and ~= 1.5 epochs on the deduped Pile (which contains 207B tokens in 1 epoch).

Config files used to train these models within the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox) can be found at the `models/` directory within this repository.

We also upload the pre-tokenized data files and a script to reconstruct the dataloader as seen during training for all models. See **Reproducing Training** section for more details.

## Changelog

[April 3, 2023] We have released a new version of all Pythia models, with the following changes to our training procedure:

- All model sizes are now trained with uniform batch size of 2M tokens. Previously, the models of size 160M, 410M, and 1.4B parameters were trained with batch sizes of 4M tokens.
- We added checkpoints at initialization (step 0) and steps {1,2,4,8,16,32,64, 128,256,512} in addition to every 1000 training steps.
- Flash Attention was used in the new retrained suite. Empirically, this seems to have effected the dynamic range of model outputs in some cases, which we are investigating further.
- We remedied a minor inconsistency that existed in the original suite: all models of size 2.8B parameters or smaller had a learning rate (LR) schedule which decayed to a minimum LR of 10% the starting LR rate, but the 6.9B and 12B models all used an LR schedule which decayed to a minimum LR of 0. In the redone training runs, we rectified this inconsistency: all models now were trained with LR decaying to a minimum of 0.1× their maximum LR.
- the new `EleutherAI/pythia-1b` is trained with bf16, because in fp16 the model corrupted due to loss spikes late in training.

The old models ("V0") remain available at [https://huggingface.co/models?other=pythia_v0](https://huggingface.co/models?other=pythia_v0).

[January 20, 2023]
On January 20, 2023, we chose to rename the \textit{Pythia} model suite to better reflect including both embedding layer and unembedding layer parameters in our total parameter counts, in line with many other model suites and because we believe this convention better reflects the on-device memory usage of these models. See [https://huggingface.co/EleutherAI/pythia-410m-deduped#naming-convention-and-parameter-count](https://huggingface.co/EleutherAI/pythia-410m-deduped#naming-convention-and-parameter-count) for more details

## Quickstart

All Pythia models are hosted on [the Huggingface hub](https://huggingface.co/EleutherAI). They can be loaded and used via the following code (shown for the 3rd `pythia-70M-deduped` model checkpoint):

```python
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
```

All models were trained for the equivalent of 143000 steps at a batch size of 2,097,152 tokens. Revision/branch `step143000` (e.g. [https://huggingface.co/EleutherAI/pythia-70m-deduped/tree/step143000](https://huggingface.co/EleutherAI/pythia-19m-deduped/tree/step143000)) corresponds exactly to the model checkpoint on the `main` branch of each model.

We additionally have all model checkpoints in the format accepted by the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox), but do not serve them at scale due to size of optimizer states and anticipated lower demand. If you would like to perform analysis using the models within the GPT-NeoX codebase, or would like the optimizer states, please email hailey@eleuther.ai and stella@eleuther.ai to arrange access.


*`pythia-{size}-v0` models on Huggingface of sizes `160m, 410m, 1.4b` were trained with a batch size of 4M tokens  and were originally trained for 71500 steps instead, and checkpointed every 500 steps. The checkpoints on Huggingface for these v0 models are renamed for consistency with all 2M batch models, so `step1000` is the first checkpoint for `pythia-1.4b-v0` that was saved (corresponding to step 500 in training), and `step1000` is likewise the first pythia-6.9b-v0 checkpoint that was saved (corresponding to 1000 "actual" steps.)*

## Reproducing Training

We provide the training data for replication of our training runs. The [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox) requires the pre-tokenized training data in the form of 2 memory-mapped numpy arrays: a `.bin` and `.idx` file.

We provide these files, hosted on the Hugging Face hub.

To download and use the deduplicated Pile training data, run:

```bash
git lfs clone https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps

python utils/unshard_memmap.py --input_file ./pythia_deduped_pile_idxmaps/pile_0.87_deduped_text_document-00000-of-00082.bin --num_shards 83 --output_dir ./pythia_pile_idxmaps/
```
This will take over a day to run, though it should not require more than 5 GB of RAM. We recommend downloading this rather than retokenizing the Pile from scratch, in order to preserve the data order seen by the Pythia models.

TODO: forthcoming: more information on how to replicate + relaunch the Pythia training runs, once the data is actually downloaded.


### Dataset Viewer

We provide a tool to view particular portions of the training dataloader used by all models during training, at `utils/batch_viewer.py`.

To run, first substitute the filepath to the downloaded `.bin` and `.idx` files for either the Pile or deduplicated Pile in `utils/dummy_config.yml`.

```python
PYTHONPATH=utils/gpt-neox/ python utils/batch_viewer.py \
  --start_iteration 0 \
  --end_iteration 1000 \
  --mode save \
  --conf_dir utils/dummy_config.yml 
```

Passing `--mode save` will save a separate file containing each batch as a numpy array. 

Passing `--mode custom` will save a dictionary for each batch to a JSONL file--it can be used to compute arbitrary statistics over each batch seen during training.


## Benchmark Scores

We also provide benchmark 0-shot and 5-shot results on a variety of NLP datasets:

- Lambada (`lambada_openai`)
- Wikitext (`wikitext`)
- PiQA (`piqa`)
- SciQ (`sciq`)
- WSC (`wsc`)
- Winogrande (`winogrande`)
- ARC-challenge (`arc_challenge`)
- ARC-easy (`arc_easy`)
- LogiQA (`logiqa`)
- BLiMP (`blimp_*`)
- MMLU (`hendrycksTest*`)

Evaluations were performed in GPT-NeoX using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), and are viewable by model and step at `results/json/v1.1-evals/*` in this repository.



## Citation Details

If you use the Pythia models or data in your research, please consider citing our paper via:

```
@misc{biderman2023pythia,
      title={Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling}, 
      author={Stella Biderman and Hailey Schoelkopf and Quentin Anthony and Herbie Bradley and Kyle O'Brien and Eric Hallahan and Mohammad Aflah Khan and Shivanshu Purohit and USVSN Sai Prashanth and Edward Raff and Aviya Skowron and Lintang Sutawika and Oskar van der Wal},
      year={2023},
      eprint={2304.01373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

```
   Copyright 2023 EleutherAI

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
