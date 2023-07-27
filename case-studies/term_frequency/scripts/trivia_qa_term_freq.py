import os
import ast
import json
import string
import argparse

import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

# few_shot_list = [16, 8, 4, 2, 0]
few_shot_list = [4]

# eval_steps, max_steps = 13_000, 143_000
# checkpoint_list = list(range(eval_steps, max_steps+eval_steps, eval_steps))
# show_checkpoints = list(range(eval_steps, max_steps+eval_steps, 2*eval_steps))
show_checkpoints = [1000, 2000, 4000]


model_list = [
    ("12 B", "EleutherAI/pythia-v1.1-12b-deduped"),
    # ("6.9 B", "EleutherAI/pythia-v1.1-6.9b-deduped"),
    ("2.8 B", "EleutherAI/pythia-v1.1-2.8b-deduped"),
    # ("1.4 B", "EleutherAI/pythia-v1.1-1.4b-deduped"),
    ("1.0 B", "EleutherAI/pythia-v1.1-1b-deduped"),
    # ("410 M", "EleutherAI/pythia-v1.1-410m-deduped"),
    ("160 M", "EleutherAI/pythia-v1.1-160m-deduped"),
    # ("70 M", "EleutherAI/pythia-v1.1-70m-deduped"),
]

def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    trivia_qa_dataset = datasets.load_dataset("trivia_qa", "unfiltered.nocontext")

    all_accuracy = []
    all_count = []
    all_checkpoint = []
    all_model = []
    for split in ["train", "validation"]:

        dataset_split = trivia_qa_dataset[split]

        for idx, (size, model) in enumerate(model_list):

            model_size = model.split("/")[-1]

            for i, checkpoint in enumerate(show_checkpoints):
                results_path = "/fsx/lintangsutawika/01-project-pythia/pythia/results/json/{}/trivia_qaterm_frequency-{}-{}-04shot.json".format(model_size, model_size, checkpoint)

                # Open model predictions
                with open(results_path, "r") as file:
                    results = json.load(file)['results']['long_tail_trivia_qa']
                    acc = results['acc']
                    question_id = results['id']

                count_path = "/fsx/lintangsutawika/01-project-pythia/data/{}/qa_co_occurrence_split={}.json".format(checkpoint, split)
                # count_path = "/fsx/lintangsutawika/01-project-pythia/data/{}/qa_co_occurrence_split={}.json".format("wikipedia", split)
                # Open entity count per each qa-pair
                doc_counts = []
                with open(count_path, 'r') as file:
                    read_count = np.asarray(ast.literal_eval(file.readlines()[0]))
                    doc_counts += read_count

                # Go through dataset to get acc - count
                _accuracy = []
                _count = []
                for sample, count in tqdm(zip(dataset_split, doc_counts), total=len(doc_counts)):
                    index = question_id.index(sample["question_id"])
                    sample_accuracy = acc[index]

                    _accuracy.append(sample_accuracy)
                    _count.append(count)
                
                all_accuracy.extend(_accuracy)
                all_count.extend(_count)
                all_checkpoint.extend([checkpoint]*len(_count))
                all_model.extend([model_size]*len(_count))

    data = pd.DataFrame(
        data={
            "acc": all_accuracy,
            "count": all_count,
            "checkpoint": all_checkpoint,
            "model": all_model
        }
    )

    bins = np.logspace(0, 6, 7)
    # bins = np.sqrt(np.logspace(-1, 6, 8)*np.logspace(0, 7, 8)) 
    data['bin'] = pd.cut(data['count'], bins=bins, labels=bins[:-1]).astype(float)
    data['bin'].fillna(1, inplace=True)
    data['bin'] = data['bin']*np.sqrt(10) # Midpoint in log-space

    sns.set_style("white")
    sns.set_context("poster")
    fig, axes = plt.subplots(1, len(model_list), figsize=(20, 7.5), tight_layout=True)

    for idx, (size, model) in enumerate(list(reversed(model_list))):

        alphabet = string.ascii_lowercase[idx]
        name = model.split("/")[-1]
        ax = axes[idx]

        model_data = data[data['model'] == name].groupby(['bin', 'checkpoint']).mean().reset_index()

        sns.lineplot(
            ax=ax,
            data=model_data,
            x='bin',
            y='acc',
            hue='checkpoint',
            palette='tab10',
            marker="o",
            errorbar=None,
            legend=False if idx != (len(model_list)-1) else True
        )

        if idx == len(model_list)-1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

        if idx != 0:
            ax.tick_params(labelleft=False, left=False)

        ax.grid()
        ax.set(xscale="log")
        ax.set(xlim=(1, 10**6))
        ax.set(ylim=(0.0, 0.55))
        ax.minorticks_off()
        ax.set_xlabel(f"\n({alphabet}) {size}")
        ax.set_ylabel("")

    prefix = "performance"
    task = "trivia_qa"
    n = 4
    figure_title = f"{task}_{n}-shot"
    plt.savefig(f'overleaf/{prefix}-{figure_title}.pdf', dpi=200)
    plt.savefig(f'overleaf/{prefix}-{figure_title}.png', dpi=200)
    plt.clf()