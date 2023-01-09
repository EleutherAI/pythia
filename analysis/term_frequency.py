import re
import os

import ast
import json

import pandas as pd

import lm_eval.models
from lm_eval import tasks, evaluator

# device = "cuda"
device = "0"
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

task_names = [
    "numerical_reasoning_arithmetic_multiplication",
    "numerical_reasoning_arithmetic_addition",
    "numerical_reasoning_op_infer_mult",
    "numerical_reasoning_op_infer_add",
    "numerical_reasoning_convert_min_sec",
    "numerical_reasoning_convert_hour_min",
    "numerical_reasoning_convert_day_hour",
    "numerical_reasoning_convert_week_day",
    "numerical_reasoning_convert_month_week",
    "numerical_reasoning_convert_year_month",
    "numerical_reasoning_convert_decade_year",
    ]


def evaluate_num_reasoning(model_name, device, batch_size=64):

    all_results_df = pd.DataFrame()
    for checkpoint in checkpoint_list:
        revision = "step{}".format(checkpoint)
        model_args="pretrained={},revision={}".format(model_name, revision)
        print("Building Model")
        model = lm_eval.models.get_model("gptneo").create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
        for n in few_shot_list:
            print("processing {} at step: {} with batch size {} and {}-shots".format(model_name, checkpoint, batch_size, n))

            results = evaluator.simple_evaluate(
                model=model,
                # model="gptneo",
                # model_args="pretrained={},revision={}".format(model_name, revision),
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

    # from multiprocessing import Process

    # for idx, (model_name, batch_size) in enumerate(model_list):
    #     p = Process(target=evaluate_num_reasoning, args=(model_name, str(idx), batch_size,))
    #     p.daemon = True
    #     p.start()

    # import argparse

    # parser = argparse.ArgumentParser(description='Eval on Num Reasoning')
    # parser.add_argument('--model_name')
    # parser.add_argument('--device')
    # parser.add_argument('--batch_size', type=int)
    # args = parser.parse_args()

    for idx, (model_name, batch_size) in enumerate(model_list):
        evaluate_num_reasoning(
            model_name,
            "cuda",
            batch_size
            )
