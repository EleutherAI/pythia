# Pythia: Interpreting Autoregressive Transformers Across Time and Scale

This repository is for EleutherAI's work-in-progress project *Pythia* which combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers.

## Models

| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| ----------- | -------- |------------ | ------- |----------- | ---------- |------------ |
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|
| Params      | n_layers |d_model      | n_heads |d_head      | Batch Size |Learning Rate|

## Experiments 

### Grammar Learning Trajectories of Language Models

### Deep and Shallow Knowledge Extraction

### Training Order and Memorization

A common explanation for language model training dynamics is that LMs have a mass of knowledge and when they come across new information they glom that knowledge on and slowly integrate it into the mass over time. One prediction that this mental model makes is that tokens encountered later in training will be more likely to be memorized than ones encountered earlier in training, as the model will not have time to adjust its representations to store the info without memorization. The primary goal of this experiment is to **disprove** this prediction.

### Social Bias and Knowledge of Social Bias
