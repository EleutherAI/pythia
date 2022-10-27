# Pythia: Interpreting Autoregressive Transformers Across Time and Scale

This repository is for EleutherAI's work-in-progress project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers.

## Models

| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate| Checkpoints | Evaluations|
| ----------- | -------- |------------ | ------- |----------- | ---------- |------------ | ---------- | --------------- |
| 19M         | 6        | 512         | 8       | 64         | 2M         | 1e-3        | ---------- | --------------- |
| 19M Dedup   | 6        | 512         | 8       | 64         | 2M         | 1e-3        | ---------- | --------------- |
| 49M         | 10       | 640         | 10      | 64         | 2M         | 8e-4        | ---------- | --------------- |
| 49M Dedup   | 10       | 640         | 10      | 64         | 2M         | 8e-4        | ---------- | --------------- |
| 125M        | 12       | 768         | 12      | 64         | 4M         | 6e-4        | ---------- | --------------- |
| 125M Dedup  | 12       | 768         | 12      | 64         | 4M         | 6e-4        | ---------- | --------------- |
| 350M        | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | ---------- | --------------- |
| 350M Dedup  | 24       | 1024        | 16      | 64         | 4M         | 3e-4        | ---------- | --------------- |
| 800M        | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | ---------- | --------------- |
| 800M Dedup  | 16       | 2048        | 8       | 128        | 4M         | 3e-4        | ---------- | --------------- |
| 1.3B        | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | ---------- | --------------- |
| 1.3B Dedup  | 24       | 2048        | 16      | 128        | 4M         | 2e-4        | ---------- | --------------- |
| 2.7B        | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | ---------- | --------------- |
| 2.7B Dedup  | 32       | 2560        | 32      | 80         | 2M         | 1.6e-4      | ---------- | --------------- |
| 6.7B        | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4 ?    | ---------- | --------------- |
| 6.7B Dedup  | 32       | 4096        | 32      | 128        | 2M         | 1.2e-4 ?    | ---------- | --------------- |
| 13B         | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | ---------- | --------------- |
| 13B Dedup   | 36       | 5120        | 40      | 128        | 2M         | 1.2e-4      | ---------- | --------------- |


`s3://pythia-hf/` contains the checkpoints that are converted to HF format.


TODO: add instructions for downloading a HF model from where they're hosted for very easy access to the intermediate ckpts

TODO: add all configs to repo




## Experiments 

### Grammar Learning Trajectories of Language Models

### Training Order and Memorization

A common explanation for language model training dynamics is that LMs have a mass of knowledge and when they come across new information they glom that knowledge on and slowly integrate it into the mass over time. One prediction that this mental model makes is that tokens encountered later in training will be more likely to be memorized than ones encountered earlier in training, as the model will not have time to adjust its representations to store the info without memorization. The primary goal of this experiment is to **disprove** this prediction.
