import re
import os

import ast
import json

import pandas as pd

from lm_eval import evaluator

from task import *
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
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, 2*eval_steps))

few_shot_list = [16, 8, 4, 2, 0]

task_names = []
task_names.extend([ArithmeticMultiplication(str(num)) for num in range(0,100)])
task_names.extend([ArithmeticAddition(str(num)) for num in range(0,100)])

def evaluate_num_reasoning(model_name, device, batch_size=64, output_dir="results/"):

    all_results_df = pd.DataFrame()
    for checkpoint in checkpoint_list:

        model_size = model_name.split("/")[-1]
        model_args="pretrained={},revision=step{}".format(model_name, checkpoint)
        print("Building Model, {}".format(model_args))
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

            results['config']['model'] = "model_name"
            results['config']['model_args'] = model_args

            dumped = json.dumps(results, indent=2)
            output_dict_dir = os.path.join(
                output_dir,
                "json",
                model_size,
                "term_frequency-{}-{}-{}shot.json".format(model_size, checkpoint, str(n).zfill(2))
            )
            with open(output_dict_dir, "w") as f:
                f.write(dumped)

            for task in task_names:
                results_dict = {
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "task": task.EVAL_HARNESS_NAME,
                    "fewshot": n,
                    **results['results'][task.EVAL_HARNESS_NAME]
                    }

                all_results_df = pd.concat(
                    [all_results_df, pd.Series(results_dict).to_frame().T],
                    ignore_index=True
                    )
        
        output_csv_path = os.path.join(output_dir, "csv", model_size)
        os.makedirs(output_csv_path, exist_ok=True)
        all_results_df.to_csv(
            os.path.join(output_csv_path, "term_frquency_all_shots.csv"),
            index=False
            )
                    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Eval on Num Reasoning')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default="results/")
    args = parser.parse_args()

    if args.model_name is not None:
        evaluate_num_reasoning(
            model_name=args.model_name,
            device=args.device if args.device is not None else device,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            )
    else:
        for idx, (model_name, batch_size) in enumerate(model_list):
            evaluate_num_reasoning(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                output_dir=args.output_dir,
                )
