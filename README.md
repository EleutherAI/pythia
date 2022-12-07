# Pythia: Interpreting Autoregressive Transformers Across Time and Scale

This repository is for EleutherAI's work-in-progress project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers.

## Models

| Params               | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate| Checkpoints | Evaluations        |
| -------------------- | -------- |------------ | ------- |----------- | ---------- |------------ | ---------- | ------------------- |
| Pythia-19M           | 6        | 512         | 8       | 64         | 2M         | 1e-3        | Ready      | Ready           |
| Pythia-19M-Deduped   | 6        | 512         | 8       | 64         | 2M         | 1e-3        | Ready      |                 |
| Pythia-125M          | 12       | 768         | 12      | 64         | 4M         | 6e-4        | Ready      | In-Progress |
| Pythia-125M-Deduped  | 12       | 768         | 12      | 64         | 4M         | 6e-4        | Ready      | --------------- |
| Pythia-350M          | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | Ready      | --------------- |
| Pythia-350M-Deduped  | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | Ready      | --------------- |
| Pythia-800M          | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | Ready      | Ready           |
| Pythia-800M-Deduped  | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | Ready      | Ready           |
| Pythia-1.3B          | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | Ready      | Ready           |
| Pythia-1.3B-Deduped  | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | Ready      | Ready           |
| Pythia-2.7B          | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | Ready      | Ready           |
| Pythia-2.7B-Deduped  | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | Ready      | Ready           |
| Pythia-6.7B          | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4 ?    | Ready      | Ready           |
| Pythia-6.7B-Deduped  | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4 ?    | Ready      | Ready           |
| Pythia-13B           | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | Ready      | --------------- |
| Pythia-13B-Deduped   | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | Ready      | --------------- |


`s3://pythia-hf/` contains the checkpoints that are converted to HF format.


TODO: add instructions for downloading a HF model from where they're hosted for very easy access to the intermediate ckpts

TODO: link to configs from table?

## Experiments 

### Grammar Learning Trajectories of Language Models

### Training Order and Memorization

A common explanation for language model training dynamics is that LMs have a mass of knowledge and when they come across new information they glom that knowledge on and slowly integrate it into the mass over time. One prediction that this mental model makes is that tokens encountered later in training will be more likely to be memorized than ones encountered earlier in training, as the model will not have time to adjust its representations to store the info without memorization. The primary goal of this experiment is to **disprove** this prediction.
