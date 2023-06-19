# Pythia Suite: Case Studies

This file describes the necessary scripts + steps to replicate our case studies performed in the Pythia paper. 

## Directory Structure

## Memorization Density Over Training

See `pythia/predictable-memorization` for more detail on how memorization over the entire Pile corpus was assessed, and `pythia/predictable-memorization/eda.ipynb` for replication of the Q-Q test and plot featured in our paper.

## Gendered Pronoun Frequency Intervention

We provide instructions on how to replicate our gendered-pronoun intervention on the training dataset here.

TODO: document the NeoX branch intervention was done in

We provide the resulting intervened models on Huggingface under `EleutherAI/pythia-intervention-MODELSIZE-deduped` on Huggingface for replication 

Evaluations can be replicated via the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (TODO: push winobias prompting code in this repo) and plots can be replicated via the notebooks in this folder. PDF plots are available in their respective subfolders as well.

## Term Frequency Effects Over Training

Relevant files for this case study can be found in the `/term_frequency` subfolder. 

TODO: instructions for what scripts are needed to replicate the data collection + evaluations