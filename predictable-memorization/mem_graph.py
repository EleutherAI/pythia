import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models = ['19m', '125m', '350m', '800m', '1.3b', '2.7b', '6.7b', '13b']
checkpoints = ['23000', '43000', '63000', '83000', '103000', '123000', '143000']
checkpoint_names = ['23m', '44m', '65m', '85m', '105m', '126m', '146m']

memorization_results = {}
folderpath = "/fsx/orz/memorization-evals/"
for model in tqdm(models):
    for idx, checkpoint in enumerate(tqdm(checkpoints)):
        modelpath = os.path.join(folderpath, f'memorization_{model}_{checkpoint}.hdf')
        memorization_results[f'{model}-{checkpoint_names[idx]}'] = pd.read_hdf(modelpath, key="memorization")
        memorization_results[f'{model}-{checkpoint_names[idx]}'].sort_values(by='index', inplace=True)


def process_memorization_over_time(models, checkpoints):

    cm_rate_df = pd.DataFrame(
        data={
            "checkpoint": [],
            "model": [],
            "TP": [],
            "FP": [],
            "FN": [],
            "TN": [],
            "TPR": [],
            "FPR": [],
            "FNR": [],
        }
    )

    # We only consider Sequence indicies that are evaluated by all checkpoints
    max_sequence_index = 23000*1024
    for model in models:

        evals = memorization_results[f'{model}-146m']
        evals = evals[evals['index'] < max_sequence_index]
        ground_truth = evals['accuracy'] == 1
        
        for idx, checkpoint in enumerate(checkpoints):
            evals = memorization_results[f'{model}-{checkpoint}']
            evals = evals[evals['index'] < max_sequence_index]
            prediction = evals['accuracy'] == 1

            matrix = confusion_matrix(prediction, ground_truth)
            TN, FP, FN, TP = matrix.ravel()
            N = max_sequence_index

            ax = sns.heatmap(np.array([[TP/N, FP/N],[FN/N, TN/N]]), annot=True, fmt=".5%")
            # set x-axis label and ticks. 
            ax.set_xlabel("Actual Labels")
            ax.xaxis.set_ticklabels(['1', '0'])
            # set y-axis label and ticks
            ax.set_ylabel("Predicted Labels")
            ax.yaxis.set_ticklabels(['1', '0'])

            ax.set_title("Predicting Memorization of Last Checkpoint\nModel Size {}, Checkpoint {}".format(model, checkpoint))
            plt.savefig('../../results/graphs/memorization_early_checkpoint_predict_last_checkpoint/graph_{}_{}.svg'.format(model, checkpoint), dpi=300)
            plt.clf()

            cm_rate_df = pd.concat(
                [
                    cm_rate_df,
                    pd.DataFrame(
                        {
                            "checkpoint": [checkpoint],
                            "model": [model],
                            "TP": [TP],
                            "FP": [FP],
                            "FN": [FN],
                            "TN": [TN],
                            "TPR": [TP/(TP+FN)],
                            "FPR": [FP/(FP+TN)],
                            "FNR": [FN/(FN+TP)],
                        }
                    )
                ],
                ignore_index=True
            )

    return cm_rate_df

cm_rate_df = process_memorization_over_time(models, checkpoint_names)
_df = cm_rate_df[cm_rate_df['checkpoint'] != '146m']
sns.lineplot(data=_df, x="checkpoint", y="TPR", hue="model")
ax.set_title("True Positive Rate of Memorization")
plt.savefig('../../results/graphs/memorization_rates_through_time/graph_tpr.svg', dpi=300)
plt.clf()
sns.lineplot(data=_df, x="checkpoint", y="FPR", hue="model")
ax.set_title("False Positive Rate of Memorization")
plt.savefig('../../results/graphs/memorization_rates_through_time/graph_fpr.svg', dpi=300)
plt.clf()
sns.lineplot(data=_df, x="checkpoint", y="FNR", hue="model")
ax.set_title("False Negative Rate of Memorization")
plt.savefig('../../results/graphs/memorization_rates_through_time/graph_fnr.svg', dpi=300)
plt.clf()

def process_memorization_over_size(models, checkpoints):

    cm_rate_df = pd.DataFrame(
        data={
            "checkpoint": [],
            "model": [],
            "TP": [],
            "FP": [],
            "FN": [],
            "TN": [],
            "TPR": [],
            "FPR": [],
            "FNR": [],
        }
    )

    # We only consider Sequence indicies that are evaluated by all checkpoints
    max_sequence_index = 23000*1024
    for checkpoint in checkpoints:

        evals = memorization_results[f'13b-{checkpoint}']
        evals = evals[evals['index'] < max_sequence_index]
        ground_truth = evals['accuracy'] == 1
        
        for model in models:
            evals = memorization_results[f'{model}-{checkpoint}']
            evals = evals[evals['index'] < max_sequence_index]
            prediction = evals['accuracy'] == 1

            matrix = confusion_matrix(prediction, ground_truth)
            TN, FP, FN, TP = matrix.ravel()
            N = max_sequence_index

            ax = sns.heatmap(np.array([[TP/N, FP/N],[FN/N, TN/N]]), annot=True, fmt=".5%")
            # set x-axis label and ticks. 
            ax.set_xlabel("Actual Labels")
            ax.xaxis.set_ticklabels(['1', '0'])
            # set y-axis label and ticks
            ax.set_ylabel("Predicted Labels")
            ax.yaxis.set_ticklabels(['1', '0'])

            ax.set_title("Predicting Memorization of Largest Size\nModel Size {}, Checkpoint {}".format(model, checkpoint))
            plt.savefig('../../results/graphs/memorization_small_model_predict_large_model/graph_{}_{}.svg'.format(model, checkpoint), dpi=300)
            plt.clf()

            cm_rate_df = pd.concat(
                [
                    cm_rate_df,
                    pd.DataFrame(
                        {
                            "checkpoint": [checkpoint],
                            "model": [model],
                            "TP": [TP],
                            "FP": [FP],
                            "FN": [FN],
                            "TN": [TN],
                            "TPR": [TP/(TP+FN)],
                            "FPR": [FP/(FP+TN)],
                            "FNR": [FN/(FN+TP)],
                        }
                    )
                ],
                ignore_index=True
            )

    return cm_rate_df


def rate_of_memorization(models, checkpoints):

    df = pd.DataFrame(
        data={
            "checkpoint": [],
            "model": [],
            "num_memorization": [],
        }
    )

    # n_steps = ['23000', '43000', '63000', '83000', '103000', '123000', '143000']
    # We only consider Sequence indicies that are evaluated by all checkpoints
    max_sequence_index = 23000*1024
    for model in models:
        
        for idx, checkpoint in enumerate(checkpoints):

            # max_sequence_index = int(n_steps[idx]) * 1024

            evals = memorization_results[f'{model}-{checkpoint}']    
            evals = evals[evals['index'] < max_sequence_index]
            prediction = evals['accuracy'] == 1

            df = pd.concat(
                [df, 
                pd.DataFrame({
                    "checkpoint": [checkpoint],
                    "model": [model],
                    "num_memorization": [prediction.mean()*100],
                })],
            ignore_index=True
            )

    return df

df = rate_of_memorization(models, checkpoint_names)
ax = sns.lineplot(data=df, x="checkpoint", y="num_memorization", hue="model")
ax.set_title("Number of Memorized Sequences Seen from Earliest Checkpoint")
ax.set_xlabel("Checkpoint") #, fontsize=14) #, labelpad=20)
ax.set_ylabel("Memorized Sequences (%)") #, fontsize=14) #, labelpad=20)
plt.savefig('graph_num_memorized.svg', dpi=300)
plt.clf()

