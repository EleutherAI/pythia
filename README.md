# Pythia: Interpreting Transformers Across Time and Scale

This repository is for EleutherAI's project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers. For detailed info on the models, their training, and their behavior, please see our paper [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373). 

Aside from the Pythia suite, this repository also acts as a hub containing information, code, and reproducibility instructions for the following papers:
* [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/abs/2304.11158)
  * For more information, see [here](/predictable-memorization/README.md).

## Contents

- [Pythia: Interpreting Transformers Across Time and Scale](#pythia--interpreting-transformers-across-time-and-scale)
  * [Models](#models)
  * [Changelog](#changelog)
- [Using Pythia](#using-pythia)
  * [Quickstart](#quickstart)
  * [Reproducing Training](#reproducing-training)
  * [Exploring the Dataset](#exploring-the-dataset)
  * [Pythia Paper Replication](#pythia-paper-replication)
- [Benchmark Scores](#benchmark-scores)
- [Research Building on Pythia](#research-building-on-pythia)
  * [EleutherAI Projects](#eleutherai-projects)
  * [Other EleutherAI Research](#other-eleutherai-research)
  * [External Interpretability Research](#external-interpretability-research)
  * [Other Research Projects](#other-research-projects)
- [Citation Details](#citation-details)
- [License](#license)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Models

| Params              | n_layers | d_model | n_heads | d_head | Batch Size | Learning Rate | Hugging Face Checkpoints                                                |
| ------------------- | -------- | ------- | ------- | ------ | ---------- | ------------- | ---------------------------------------------------------- |
| Pythia-14M          | 6        | 128     | 4       |      | 2M         |           | [Standard](https://huggingface.co/EleutherAI/pythia-14m)  |
| Pythia-31M          | 6        | 156     | 8       |      | 2M         |           | [Standard](https://huggingface.co/EleutherAI/pythia-31m) |
| Pythia-70M          | 6        | 512     | 8       | 64     | 2M         | 1.0e-3          | [Standard](https://huggingface.co/EleutherAI/pythia-70m), [Deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)  |
| Pythia-160M         | 12       | 768     | 12      | 64     | 2M         | 6.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-160m), [Deduped](https://huggingface.co/EleutherAI/pythia-160m-deduped)|
| Pythia-410M         | 24       | 1024    | 16      | 64     | 2M         | 3.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-410m), [Deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped)|
| Pythia-1B           | 16       | 2048    | 8       | 256    | 2M         | 3.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-1b), [Deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped)    |
| Pythia-1.4B         | 24       | 2048    | 16      | 128    | 2M         | 2.0e-4          | [Standard](https://huggingface.co/EleutherAI/pythia-1.4b), [Deduped](https://huggingface.co/EleutherAI/pythia-1.4b-deduped)|
| Pythia-2.8B         | 32       | 2560    | 32      | 80     | 2M         | 1.6e-4        | [Standard](https://huggingface.co/EleutherAI/pythia-2.8b), [Deduped](https://huggingface.co/EleutherAI/pythia-2.8b-deduped)|
| Pythia-6.9B         | 32       | 4096    | 32      | 128    | 2M         | 1.2e-4        | [Standard](https://huggingface.co/EleutherAI/pythia-6.9b), [Deduped](https://huggingface.co/EleutherAI/pythia-6.9b-deduped)|
| Pythia-12B          | 36       | 5120    | 40      | 128    | 2M         | 1.2e-4        | [Standard](https://huggingface.co/EleutherAI/pythia-12b), [Deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped)  |

We train and release a suite of 8 model sizes on the the Pile ([paper](https://pile.eleuther.ai/), [datasheet](https://arxiv.org/abs/2201.07311)) as well as the Pile with deduplication applied. All 8 model sizes are trained on the exact same data, in the exact same order. Each model saw 299,892,736,000 ~= 299.9B tokens during training, and *143 checkpoints* for each model are saved every 2,097,152,000 ~= 2B tokens, evenly spaced throughout training. This corresponds to just under 1 epoch on the Pile for non-"deduped" models, and ~= 1.5 epochs on the deduped Pile (which contains 207B tokens in 1 epoch).

Config files used to train these models with the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox) can be found at the `models/` directory within this repository, as well as in the GPT-NeoX library itself.

We also upload the pre-tokenized data files and a script to reconstruct the dataloader as seen during training for all models. See [Reproducing Training](#reproducing-training) section for more details.

## Changelog

[Oct 6, 2023] We have added 14M and 31M models at the request of some researchers. We plan on training deduped versions of these models in the future.

[April 3, 2023] We have released a new version of all Pythia models, with the following changes to our training procedure:

- All model sizes are now trained with uniform batch size of 2M tokens. Previously, the models of size 160M, 410M, and 1.4B parameters were trained with batch sizes of 4M tokens.
- We added checkpoints at initialization (step 0) and steps {1,2,4,8,16,32,64, 128,256,512} in addition to every 1000 training steps.
- Flash Attention was used in the new retrained suite. Empirically, this seems to have effected the dynamic range of model outputs in some cases, which we are investigating further.
- We remedied a minor inconsistency that existed in the original suite: all models of size 2.8B parameters or smaller had a learning rate (LR) schedule which decayed to a minimum LR of 10% the starting LR rate, but the 6.9B and 12B models all used an LR schedule which decayed to a minimum LR of 0. In the redone training runs, we rectified this inconsistency: all models now were trained with LR decaying to a minimum of 0.1× their maximum LR.
- the new `EleutherAI/pythia-1b` is trained with bf16, because in fp16 the model corrupted due to loss spikes late in training.

The old models ("v0") remain available [here](https://huggingface.co/models?other=pythia_v0) and may be useful for ablation studies.

[January 20, 2023]
On January 20, 2023, we chose to rename the Pythia model suite to include both embedding layer and unembedding layer parameters in our total parameter counts, in line with many other model suites and because we believe this convention better reflects the on-device memory usage of these models. We also discovered that due to a typo one of our models was smaller than we thought, and replaced it with a model of the intended size. See [here](https://huggingface.co/EleutherAI/pythia-410m-deduped#naming-convention-and-parameter-count) for more details.

# Using Pythia

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

We additionally have all model checkpoints in the format accepted by the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox), with final-step checkpoints+optimizer states downloadable at [`EleutherAI/neox-ckpt-pythia-xxx-deduped-v1`](https://huggingface.co/EleutherAI/neox-ckpt-pythia-1b-deduped-v1) but do not serve them for all steps at scale due to size of optimizer states and anticipated lower demand. If you would like to perform analysis using the intermediate models within the GPT-NeoX codebase, or would like the optimizer states for other steps, please email hailey@eleuther.ai and stella@eleuther.ai to arrange access.


*`pythia-{size}-v0` models on Huggingface of sizes `160m, 410m, 1.4b` were trained with a batch size of 4M tokens  and were originally trained for 71500 steps instead, and checkpointed every 500 steps. The checkpoints on Huggingface for these v0 models are renamed for consistency with all 2M batch models, so `step1000` is the first checkpoint for `pythia-1.4b-v0` that was saved (corresponding to step 500 in training), and `step1000` is likewise the first pythia-6.9b-v0 checkpoint that was saved (corresponding to 1000 "actual" steps.)*

## Reproducing Training

(Expanded reproduction instructions provided by @BaruchG ).

1. We provide the training data for replication of our training runs. The [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox) requires the pre-tokenized training data in the form of 2 memory-mapped numpy arrays: a `.bin` and `.idx` file.
We provide these files, hosted on the Hugging Face hub.
To download and use the deduplicated Pile training data, run:
```bash
git lfs clone https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps

python utils/unshard_memmap.py --input_file ./pythia_deduped_pile_idxmaps/pile_0.87_deduped_text_document-00000-of-00082.bin --num_shards 83 --output_dir ./pythia_pile_idxmaps/
``` 
   This will take over a day to run, though it should not require more than 5 GB of RAM. We recommend downloading this rather than retokenizing the Pile from scratch, in order to guarantee preservation of the data order seen by the Pythia models.

2. Make a local copy of the tokenizer from the Pythia repo at https://github.com/EleutherAI/pythia/blob/main/utils/20B_tokenizer.json 

3. Run  `git clone https://github.com/EleutherAI/gpt-neox.git` to clone the GPT-NeoX library. Once inside the repo run `git checkout v1.0` to switch to the 1.0 branch which Pythia was trained with.

4. Choose the Yaml of the model that you want to reproduce from https://github.com/EleutherAI/pythia/tree/main/models . Each size model has a Yaml for the standard Pile dataset and the deduplicated one.  Make a local copy of your selected model’s yaml.

5. Build the dockerfile contained in the v1.0 by going to the root directory of your cloned GPT-NeoX repository and running `docker build -t pythia:latest .` (assuming you have docker installed).

6. After the container finishes building run the container using the following command (from the root of the GPT-NeoX repo with your pythia yaml accessible from within that folder):
```
docker run --runtime=nvidia --rm -it -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox -v $(pwd):/workspace/ pythia:latest bash
```
Use the -v argument to add more connected volumes for the dataset and the Yaml file if is not accessible from within the docker container.

7. Change the lines of the data paths and tokenizer paths as follows:
```
  "train-data-paths": ["/fsx/pile/pile_20B_tokenizer_text_document"], #point this to your folder which was generated in step 1 containing the .bin and .idx file
  "valid-data-paths": ["/fsx/pile/pile_20B_tokenizer_text_document"], #point this to your folder which was generated in step 1 containing the .bin and .idx file
  "test-data-paths": ["/fsx/pile/pile_20B_tokenizer_text_document"], #point this to your folder which was generated in step 1 containing the .bin and .idx file

  "tokenizer-type": "HFTokenizer",
  "vocab-file": "/fsx/pile/20B_tokenizer.json", # point this to the tokenizer retrieved in step 2
```
You should additionally modify the total batch size (calculated via `Total GPUs * train_micro_batch_size_per_gpu * gradient_accumulation_steps / (pipe-parallel-size * model-parallel-size)`) to be 1024 to match the Pythia training batch size.
Total GPU counts for each Pythia training run can be observed in comments in the yaml file.
```
   "train_micro_batch_size_per_gpu": XXX, # make this a value that will fit within your GPU memory
   "gradient_accumulation_steps": 1, # make this a value to compensate to make the total batch size 1024.
```

If you would like your weights to be saved add that information to the yaml file as well. For example, to save in the checkpoints folder, at the bottom you can add:
```
  "launcher": "slurm",
  "deepspeed_slurm": false,

  "save": "checkpoints",
  "load": "checkpoints",
  "checkpoint_validation_with_forward_pass": False,
}
```
Make sure the paths are the paths from inside your docker container and if you want the weights to have persistence, make sure that they are accessible from outside the container, for example in /workspace/ .

8. Pip install flash attention by running `pip install -r requirements/requirements-flashattention.txt` from within the GPT-NeoX repository root folder inside the docker container.

9. You should now be able to start training your model by running (modify the path to your yaml file):
```
python deepy.py train.py /workspace/pythia/models/70M/pythia-70m.yml  2>&1 | tee output.txt
```
the output will be saved to output.txt, if you don’t want that just delete the end.

10. Once training is completed you can then benchmark your weights if desired. The most straightforward way to do this is using EleutherAI’s LM Evalutation Harness at https://github.com/EleutherAI/lm-evaluation-harness.  
In order to use that with your saved out weights you must first convert them from GPT-NeoX format to Huggingface format.  This can be done from inside the GPT-NeoX repository with the script at tools/convert_to_hf.py.   
If you are using the v1.0 of GPT-NeoX you may have to add `from typing import List` to the type of the file and change the line at https://github.com/EleutherAI/gpt-neox/blob/71df4d5017f9f4919566a11454fe3a507ffdc632/tools/convert_to_hf.py#L44 from `list[torch.Tensor]` to `List[torch.Tensor]`.
You can then run the script like this to convert the weights at step 143000:
```
python tools/convert_to_hf.py --input_dir checkpoints/global_step143000/ --config_file checkpoints2/global_step 143000/configs/pythia-70m.yml --output_dir ./output/ 
```
This should output a file structure similar to the one found at https://huggingface.co/EleutherAI/pythia-70m-deduped/tree/main.

11. If your `tokenizer_config.json` looks different than the one at https://huggingface.co/EleutherAI/pythia-70m-deduped/blob/main/tokenizer_config.json and `special_tokens_map.json` look different than https://huggingface.co/EleutherAI/pythia-70m-deduped/blob/main/special_tokens_map.json you may need to replace them with the ones on Huggingface.  If you don’t do this some of the tests in the Harness may not work.

12. You should then be able to set up your environment for benchmarking.  The containers at https://hub.docker.com/r/huggingface/transformers-pytorch-gpu/tags should work for this and have worked with the 4.28 and 4.29 versions.  After setting up that docker container run:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
as outlined in the Harness repository.

13. You should then be able to run the benchmark by pointing it at your weights (which should be in your container) by running a command similar to this:
```
python3 main.py     --model hf-causal-experimental     --model_args pretrained=../gpt-neox/output/     --tasks lambada_openai,piqa,winogrande,arc_easy,sciq,wikitext     --device cuda:3
```
which should output your results.

## Exploring the Dataset

We provide a tool to view particular portions of the training dataloader used by all models during training, at `utils/batch_viewer.py`.

This tool requires the `inspect_idxmap` branch of GPT-NeoX as a git submodule, so you must check out the repository via
```
git clone --recurse-submodules https://github.com/EleutherAI/pythia
cd pythia
```
or, if you have already cloned the repository, run
```
git submodule update --init --recursive
```
Next, we must install dependencies:
```
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch/
cd utils/gpt-neox
pip install -r requirements/requirements.txt
```
Additionally, we are required to build C++ helpers used by the Megatron dataloader. You can do this via:
```
cd /utils/gpt-neox/megatron/data
make
cd -
```
Now, we're all set up to run `utils/batch_viewer.py` !

To run, first substitute the filepath to your copy of the downloaded and resharded `.bin` and `.idx` files for either the Pile or deduplicated Pile in `utils/dummy_config.yml`.

```python
PYTHONPATH=utils/gpt-neox/ python utils/batch_viewer.py \
  --start_iteration 0 \
  --end_iteration 1000 \
  --mode save \
  --save_path .../.../.../... \
  --conf_dir utils/dummy_config.yml 
```

Passing `--mode save` will save a separate file containing each batch as a numpy array. 

Passing `--mode custom` will save a dictionary for each batch to a JSONL file--it can be used to compute arbitrary statistics over each batch seen during training.

## Pythia Paper Replication

We provide further information for those interested in replicating our case studies performed in the Pythia suite paper, being

* Memorization density over training
* Intervention on pronoun frequencies in pretraining
* Term frequency effects over training

# Benchmark Scores

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

Evaluations were performed in GPT-NeoX using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), and are viewable by model and step at `evals/pythia-v1/*/*` in this repository.

# Research Building on Pythia

Our primary goal with the Pythia project is to enable research on interpretability and learning dynamics at EleutherAI and in the community writ large. Here we document select papers using our models. For a complete list of papers citing Pythia, see [here](https://www.semanticscholar.org/paper/Pythia%3A-A-Suite-for-Analyzing-Large-Language-Models-Biderman-Schoelkopf/be55e8ec4213868db08f2c3168ae666001bea4b8#citing-papers).

Citation information for other papers in this repository are included in their respective folders.

## Other EleutherAI Research

- Gupta, et al. "[Continual Pre-Training of Large Language Models: How to re-warm your model?](https://arxiv.org/abs/2308.04014)." _Workshop on Efficient Systems for Foundation Models @ ICML_. 2023.

## External Interpretability Research

- Large Language Models Sometimes Generate Purely Negatively-Reinforced Text

## Other Research Projects

- OpenAssistant Conversations--Democratizing Large Language Model Alignment

# Citation Details

If you use the Pythia models or data in your research, please cite our paper via:

```
@inproceedings{biderman2023pythia,
  title={Pythia: A suite for analyzing large language models across training and scaling},
  author={Biderman, Stella and Schoelkopf, Hailey and Anthony, Quentin Gregory and Bradley, Herbie and O’Brien, Kyle and Hallahan, Eric and Khan, Mohammad Aflah and Purohit, Shivanshu and Prashanth, USVSN Sai and Raff, Edward and others},
  booktitle={International Conference on Machine Learning},
  pages={2397--2430},
  year={2023},
  organization={PMLR}
}

```

# License
The following license applies to all code in this GitHub repo, as well as the Pythia models and any other copyrightable artifacts

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
