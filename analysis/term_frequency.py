import os
import json

from lm_eval import tasks, evaluator

model_list = [
    "EleutherAI/pythia-19m-deduped",
    "EleutherAI/pythia-125m-deduped",
    "EleutherAI/pythia-350m-deduped",
    ]

eval_steps, max_steps = 13_000, 143_000
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, eval_steps))

few_shot_list = [0,2,4,8,16]

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

device = "cpu"

for model in model_list:
    for checkpoint in checkpoint_list:

        revision = "step{}".format(checkpoint)

        for n in few_shot_list:

            results = evaluator.simple_evaluate(
                model="gpt2",
                model_args="pretrained={},revision={}".format(model, revision),
                tasks=task_names,
                num_fewshot=n,
                batch_size=1,
                device=device,
                # no_cache=False,
            )

            dumped = json.dumps(results, indent=2)
            output_path = "output_{}_{}_{}-shot.csv".format(model, checkpoint, n)
            with open(output_path, "w") as f:
                f.write(dumped)