import re
import os

import ast
import json

import pandas as pd

from lm_eval import evaluator

from task import *
from trivia_qa import TriviaQA
from model import GPTNeoLM

device = "cuda"
model_list = [
    # ["EleutherAI/pythia-v1.1-160m-deduped", 64],
    ["EleutherAI/pythia-v1.1-1b-deduped", 64],
    ["EleutherAI/pythia-v1.1-2.8b-deduped", 32],
    ["EleutherAI/pythia-v1.1-12b-deduped", 8],
    ]

eval_steps, max_steps = 13_000, 143_000
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, 2*eval_steps))

def evaluate_task_performance(model_name, task_names, device, checkpoint_list=checkpoint_list, few_shot_list=[16, 8, 4, 2, 0], batch_size=64, output_dir="results/", file_name_affix=""):

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
            output_dict_path = os.path.join(
                output_dir,
                "json",
                model_size,
            )

            dict_file_name = "{}term_frequency-{}-{}-{}shot.json".format(file_name_affix, model_size, checkpoint, str(n).zfill(2))
            os.makedirs(output_dict_path, exist_ok=True)
            with open(os.path.join(output_dict_path, dict_file_name), "w") as f:
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
            os.path.join(output_csv_path, "{}term_frquency_all_shots.csv".format(file_name_affix)),
            index=False
            )
                    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Eval on Num Reasoning')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--few_shot_list', type=str, default=None)
    parser.add_argument('--checkpoint_list', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="results/")
    parser.add_argument('--task_names', type=str, default=None)
    parser.add_argument('--file_name_affix', type=str, default="")
    args = parser.parse_args()

    if args.task_names is None:
        arithmetic_task_names = []
        arithmetic_task_names.extend([ArithmeticMultiplication(str(num)) for num in range(0,100)])
        arithmetic_task_names.extend([ArithmeticAddition(str(num)) for num in range(0,100)])
        task_names = arithmetic_task_names
    else:
        # task_names = list(args.task_names.split(","))
        task_names = [
            TriviaQA(),
            # NaturalQs()
            ]

    if args.checkpoint_list is None:
        checkpoint_list = checkpoint_list
    else:
        checkpoint_list = [int(n) for n in args.checkpoint_list.split(",")]

    if args.few_shot_list is None:
        few_shot_list = [16,8,4,2,0]
    else:
        few_shot_list = [int(n) for n in args.few_shot_list.split(",")]

    if args.model_name is not None:
        evaluate_task_performance(
            model_name=args.model_name,
            task_names=task_names,
            device=args.device if args.device is not None else device,
            batch_size=args.batch_size,
            checkpoint_list=checkpoint_list,
            few_shot_list=few_shot_list,
            output_dir=args.output_dir,
            file_name_affix=args.file_name_affix,
            )
    else:
        for idx, (model_name, batch_size) in enumerate(model_list):
            evaluate_task_performance(
                model_name=model_name,
                task_names=task_names,
                device=device,
                batch_size=batch_size,
                checkpoint_list=checkpoint_list,
                few_shot_list=few_shot_list,
                output_dir=args.output_dir,
                file_name_affix=args.file_name_affix,
                )
