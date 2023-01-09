import os

import numpy as np
import pandas as pd
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


def confusion_matrix_predicting_late_memorization(models, checkpoints):

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
        ground_truth_146m = evals['accuracy'] == 1
        
        for idx, checkpoint in enumerate(checkpoints):
            evals = memorization_results[f'{model}-{checkpoint}']
            evals = evals[evals['index'] < max_sequence_index]
            prediction = evals['accuracy'] == 1
        
            matrix = confusion_matrix(prediction, ground_truth_146m, labels=[1,0])
            matrix = matrix.transpose()

            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[1,0])
            disp.plot()
            title = f"{model} Memorization Confusion Matrix"
            plt.title(title)
            plt.savefig('predicting_late_memorization_{}_{}.png'.format(model, checkpoint))

            TP = matrix[0][0]
            FP = matrix[0][1]
            FN = matrix[1][0]
            TN = matrix[1][1]

            cm_rate_df = cm_rate_df.append(
                {
                    "checkpoint": checkpoint,
                    "model": model,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "TPR": TP/(TP+FN),
                    "FPR": FP/(FP+TN),
                    "FNR": FN/(FN+TP),
                },
                ignore_index=True
            )

    return cm_rate_df

df = confusion_matrix_predicting_late_memorization(models, checkpoint_names[:-1])

def forgetting_memorization_through_time(models, checkpoints):

    df = pd.DataFrame(
        data={
            "checkpoint": [],
            "model": [],
            "num_memorization": [],
        }
    )

    # We only consider Sequence indicies that are evaluated by all checkpoints
    max_sequence_index = 23000*1024
    for model in models:
        
        for idx, checkpoint in enumerate(checkpoints):
            evals = memorization_results[f'{model}-{checkpoint}']    
            evals = evals[evals['index'] < max_sequence_index]

            if idx == 0:
                base_index = evals[evals['accuracy'] == 1].index
            
            prediction = evals.iloc[base_index]['accuracy'].sum()

            df = df.append(
                {
                    "checkpoint": checkpoint,
                    "model": model,
                    "num_memorization": prediction,
                },
                ignore_index=True
            )
    return df

df = forgetting_memorization_through_time(models, checkpoint_names)
df = df.pivot("model", "checkpoint", "num_memorization")
df = df.reindex(index=models, columns=checkpoints)
sns.heatmap(df, cmap="crest", annot=True, fmt=".1f")
plt.title("Number of Memorized Lines")
plt.savefig('Num_Mem.png')

