import re
import os
import sys
import string

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter

# few_shot_list = [16, 8, 4, 2, 0]
few_shot_list = [0, 2, 4, 8, 16]

eval_steps, max_steps = 13_000, 143_000
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, eval_steps))
show_checkpoints = list(range(eval_steps, max_steps+eval_steps, 2*eval_steps))

freq = {}
_freq = Counter()
for idx, checkpoint in enumerate(checkpoint_list):

    checkpoint_path = os.path.join("results", f"frequency_count_checkpoint_{checkpoint}.npy")
    _dict = np.load(checkpoint_path, allow_pickle=True)[()]
    _freq.update(_dict)
    freq[checkpoint] = _freq.copy()

def jitter(values, j=0):
    return values + np.random.normal(0, 0.05, values.shape)

model_list = [
    ("12 B", "EleutherAI/pythia-13b-deduped"),
    ("6.9 B", "EleutherAI/pythia-6.7b-deduped"),
    ("2.8 B", "EleutherAI/pythia-2.7b-deduped"),
    ("1.4 B", "EleutherAI/pythia-1.3b-deduped"),
    ("1.0 B", "EleutherAI/pythia-800m-deduped"),
    ("410 M", "EleutherAI/pythia-350m-deduped"),
    ("160 M", "EleutherAI/pythia-125m-deduped"),
    ("70 M", "EleutherAI/pythia-19m-deduped"),
]

sns.set_style("white")
sns.set_context("poster")

task_names = [
    "num_reasoning_arithmetic_multiplication",
    "num_reasoning_arithmetic_addition",
    # "num_reasoning_op_infer_multiplication",
    # "num_reasoning_op_infer_addition",
    ]

def graph_plot(model_list, single_row=True, prefix="figure"):

    for task in tqdm(task_names):

        if single_row:
            fig, axes = plt.subplots(1, len(model_list), figsize=(20, 7.5), tight_layout=True)
        else:
            fig, axes = plt.subplots(2, 4, figsize=(20, 15), tight_layout=True)

        for idx, (size, model) in enumerate(list(reversed(model_list))):

            alphabet = string.ascii_lowercase[idx]
            name = model.split("/")[-1]

            n = 16

            x, y, z, c = [], [], [], []
            for checkpoint in show_checkpoints:

                _freq = freq[checkpoint]
                df = pd.read_csv(f"results/csv/{name}/term_frquency_all_shots.csv")
                df = df[df["task"].str.contains(task) & (df["checkpoint"] == checkpoint) & (df["fewshot"] == n)]

                for i in range(0, 100):
                    count = str(i)
                    x.append(_freq[count])
                    y.append(df[df['task'].str.contains("_"+count)]['acc'].values[0])
                    z.append(str(checkpoint))

            data = pd.DataFrame(
                data={
                    'x': x,
                    'y': y,
                    'checkpoint': z,
                    })

            bins = np.logspace(5, np.log10(2*10**8), 20)
            data['bin'] = pd.cut(data['x'], bins=bins, labels=bins[:-1]).astype(int)

            if single_row:
                ax = axes[idx]
            else:
                x_idx = idx//4
                y_idx = idx%4
                ax = axes[x_idx, y_idx]

            sns.lineplot(
                ax=ax,
                data=data,
                x='bin',
                y='y',
                hue='checkpoint',
                marker="o",
                errorbar=None,
                legend=False if idx != (len(model_list)-1) else True
            )

            if idx == len(model_list)-1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            if single_row:
                if idx != 0:
                    ax.tick_params(labelleft=False, left=False)
            else:
                if y_idx != 0:
                    ax.tick_params(labelleft=False, left=False)

            ax.set(xscale="log")
            ax.set(ylim=(0.0, 1.0))
            ax.set_xlabel(f"\n({alphabet}) {size}")
            ax.set_ylabel("")

        figure_title = f"{task}_{n}-shot"
        plt.savefig(f'overleaf/{prefix}-{figure_title}.pdf', dpi=200)
        plt.savefig(f'overleaf/{prefix}-{figure_title}.png', dpi=200)
        plt.clf()

graph_plot(
    [
        ("12 B", "EleutherAI/pythia-13b-deduped"),
        ("6.9 B", "EleutherAI/pythia-6.7b-deduped"),
        ("2.8 B", "EleutherAI/pythia-2.7b-deduped"),
        ("1.4 B", "EleutherAI/pythia-1.3b-deduped"),
        ("1.0 B", "EleutherAI/pythia-800m-deduped"),
        ("410 M", "EleutherAI/pythia-350m-deduped"),
        ("160 M", "EleutherAI/pythia-125m-deduped"),
        ("70 M", "EleutherAI/pythia-19m-deduped"),
    ],
    single_row=False,
    prefix="appendix",
    )

graph_plot(
    [
        ("12 B", "EleutherAI/pythia-13b-deduped"),
        ("2.8 B", "EleutherAI/pythia-2.7b-deduped"),
        ("1.0 B", "EleutherAI/pythia-800m-deduped"),
        ("160 M", "EleutherAI/pythia-125m-deduped"),
    ]
    )

few_shot_list = [0,4,16]
for task in tqdm(task_names):

    mux = pd.MultiIndex.from_product([
        [i[0] for i in reversed(model_list)],
        few_shot_list
    ])
    table = pd.DataFrame(
        data={
            key: [np.nan]*len(show_checkpoints) for key in mux
            },
        index=show_checkpoints,
        columns=mux
    )

    for idx, (size, model) in enumerate(list(reversed(model_list))):

        alphabet = string.ascii_lowercase[idx]
        name = model.split("/")[-1]

        # for n in few_shot_list:
        for n in [0,4,16]:

            for checkpoint in show_checkpoints:

                _freq = freq[checkpoint]
                # df = pd.read_csv(f"results/csv/pythia-{size}-deduped/term_frquency_all_shots.csv")
                df = pd.read_csv(f"results/csv/{name}/term_frquency_all_shots.csv")
                df = df[df["task"].str.contains(task) & (df["checkpoint"] == checkpoint) & (df["fewshot"] == n)]
                # df = df[df["task"].str.contains(task) & (df["fewshot"] == n)]

                x, y, c = [], [], []
                for i in range(0, 100):
                    count = str(i)
                    c.append(_freq[count])
                    x.append(i)
                    y.append(df[df['task'].str.contains("_"+count)]['acc'].values[0])

                data = pd.DataFrame(
                    data={
                        'x': x,
                        'y': y,
                        'c': c,
                        })
                data.sort_values(by='c', ascending=False, inplace=True)

                top_10p = data['y'].iloc[:10].mean()
                bottom_10p = data['y'].iloc[-10:].mean()
                delta_performance = top_10p - bottom_10p
                table[(size, n)].loc[checkpoint] = delta_performance*100

    print(table.to_latex())