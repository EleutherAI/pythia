# Pythia: Interpreting Autoregressive Transformers Across Time and Scale

This repository is for EleutherAI's work-in-progress project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers. 

## Models

| Params               | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate| Checkpoints | Evaluations        |
| -------------------- | -------- |------------ | ------- |----------- | ---------- |------------ | ---------- | ------------------- |
| Pythia-19M           | 6        | 512         | 8       | 64         | 2M         | 1e-3        | [Here](https://huggingface.co/EleutherAI/pythia-19m)      | Ready           |
| Pythia-19M-Deduped   | 6        | 512         | 8       | 64         | 2M         | 1e-3        | [Here](https://huggingface.co/EleutherAI/pythia-19m-deduped)     | Ready           |
| Pythia-125M          | 12       | 768         | 12      | 64         | 4M         | 6e-4        | [Here](https://huggingface.co/EleutherAI/pythia-125m)      | Ready |
| Pythia-125M-Deduped  | 12       | 768         | 12      | 64         | 4M         | 6e-4        | [Here](https://huggingface.co/EleutherAI/pythia-125m-deduped)      | --------------- |
| Pythia-350M          | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | [Here](https://huggingface.co/EleutherAI/pythia-350m)     | --------------- |
| Pythia-350M-Deduped  | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | [Here](https://huggingface.co/EleutherAI/pythia-350m-deduped)      | --------------- |
| Pythia-800M          | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | [Here](https://huggingface.co/EleutherAI/pythia-800m)      | Ready           |
| Pythia-800M-Deduped  | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | [Here](https://huggingface.co/EleutherAI/pythia-800m-deduped)      | Ready           |
| Pythia-1.3B          | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-1.3b)      | Ready           |
| Pythia-1.3B-Deduped  | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | [Here](https://huggingface.co/EleutherAI/pythia-1.3b-deduped)      | Ready           |
| Pythia-2.7B          | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | [Here](https://huggingface.co/EleutherAI/pythia-2.7b)      | Ready           |
| Pythia-2.7B-Deduped  | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | [Here](https://huggingface.co/EleutherAI/pythia-2.7b-deduped)      | Ready           |
| Pythia-6.7B          | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4      | [Here](https://huggingface.co/EleutherAI/pythia-6.7b)      | Ready           |
| Pythia-6.7B-Deduped  | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4      | [Here](https://huggingface.co/EleutherAI/pythia-6.7b-deduped)      | Ready           |
| Pythia-13B           | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | [Here](https://huggingface.co/EleutherAI/pythia-13b)      | --------------- |
| Pythia-13B-Deduped   | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | [Here](https://huggingface.co/EleutherAI/pythia-13b-deduped)      | --------------- |

We train and release a suite of 8 model sizes on 2 different datasets: [the Pile](https://pile.eleuther.ai/), as well as the Pile with deduplication applied.

All 8 model sizes are trained on the exact same data, in the exact same order. Each model saw 299,892,736,000 ~= 299.9B tokens during training, and *143 checkpoints* for each model are saved every 2,097,152,000 ~= 2B tokens, evenly spaced throughout training. This corresponds to just under 1 epoch on the Pile for non-"deduped" models, and ~= 1.5 epochs on the deduped Pile (which contains 207B tokens in 1 epoch).

Config files used to train these models within the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox) can be found at the `models/` directory within this repository.

We are planning on releasing a user-friendly utility, but the dataset loader in GPT-NeoX (which is the same as the Megatron-DS one, AFAIK) allows you to save and export the seed used to shuffle data. If you just build the GPT2Dataset with the right neox args (provided with the model checkpoints) then it should be easily accessible — just grab the `[(BS * starting iter) + index]`th element of the dataset object.

## Quickstart

All Pythia models are hosted on [the Huggingface hub](https://huggingface.co/EleutherAI). They can be loaded and used via the following code (shown for the 3rd `pythia-19M-deduped` model checkpoint):

```python
from transformers import GPTNeoXForCausalLM

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-19m-deduped",
  revision="step3000",
  cache_dir="./pythia-19m-deduped/step3000",
)
  
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-19m-deduped",
  revision="step3000",
  cache_dir="./pythia-19m-deduped/step3000",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
model.generate(**inputs)
```

All models were trained for the equivalent of 143000 steps at a batch size of 2,097,152 tokens. Revision/branch `step143000` (e.g. [https://huggingface.co/EleutherAI/pythia-19m-deduped/tree/step143000](https://huggingface.co/EleutherAI/pythia-19m-deduped/tree/step143000)) corresponds exactly to the model checkpoint on the `main` branch of each model.
 
We additionally have all model checkpoints in the format accepted by the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox), but do not serve them at scale due to size of optimizer states and anticipated lower demand. If you would like to perform analysis using the models within the GPT-NeoX codebase, or would like the optimizer states, please email us at stella@eleuther.ai to arrange access.

## Experiments 

### Grammar Learning Trajectories of Language Models

### Training Order and Memorization

A common explanation for language model training dynamics is that LMs have a mass of knowledge and when they come across new information they glom that knowledge on and slowly integrate it into the mass over time. One prediction that this mental model makes is that tokens encountered later in training will be more likely to be memorized than ones encountered earlier in training, as the model will not have time to adjust its representations to store the info without memorization. The primary goal of this experiment is to **disprove** this prediction and demonstrate that training order doesn't influence memorization.

### Grammar Learning Trajectories of Language Models


