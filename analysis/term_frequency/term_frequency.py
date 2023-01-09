import re
import os

import ast
import json

import pandas as pd

import lm_eval.models
from lm_eval import tasks, evaluator

from task import ArithmeticMultiplication
from model import GPTNeoLM

device = "cuda"
model_list = [
    ["EleutherAI/pythia-13b-deduped", 32],
    ["EleutherAI/pythia-6.7b-deduped", 32],
    ["EleutherAI/pythia-2.7b-deduped", 64],
    ["EleutherAI/pythia-1.3b-deduped", 64],
    ["EleutherAI/pythia-800m-deduped", 128],
    ["EleutherAI/pythia-350m-deduped", 128],
    ["EleutherAI/pythia-125m-deduped", 256],
    ["EleutherAI/pythia-19m-deduped", 256],
    ]

eval_steps, max_steps = 13_000, 143_000
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, eval_steps))

few_shot_list = [16, 8, 4, 2, 0]

from task import *

task_names = []
task_names.extend([ArithmeticMultiplication(str(num)) for num in range(0,100)])
task_names.extend([ArithmeticAddition(str(num)) for num in range(0,100)])
task_names.extend([OperationInferenceMult(str(num)) for num in range(0,100)])
task_names.extend([OperationInferenceAdd(str(num)) for num in range(0,100)])

def evaluate_num_reasoning(model_name, device, batch_size=64):

    all_results_df = pd.DataFrame()
    for checkpoint in checkpoint_list:

        print("Building Model")
        model_args="pretrained={},revision=step{}".format(model_name, checkpoint)
        model = GPTNeoLM.create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )

        for n in few_shot_list:
            print("processing {} at step: {} with batch size {} and {}-shots".format(model_name, checkpoint, batch_size, n))

            results = evaluator.simple_evaluate(
                model=model,
                tasks=task_names,
                num_fewshot=n,
                batch_size=batch_size,
                device=device,
                no_cache=True,
            )

            for task in task_names:
                results_dict = {
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "task": task,
                    "fewshot": n,
                    **results['results'][task]
                    }

                all_results_df = pd.concat(
                    [all_results_df, pd.Series(results_dict).to_frame().T],
                    ignore_index=True
                    )
        
        all_results_df.to_csv(
            "lm_cache/{}_numerical_reasoning_through_time.csv".format(
                re.sub("/", "_", model_name)),
                index=False
            )
                    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Eval on Num Reasoning')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    for idx, (model_name, batch_size) in enumerate(model_list):

        evaluate_num_reasoning(
            model_name,
            device,
            args.batch_size if args.batch_size is not None else batch_size
            )
