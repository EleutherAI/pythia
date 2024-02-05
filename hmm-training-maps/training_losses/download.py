import pandas as pd
import wandb
from tqdm.auto import tqdm
from pathlib import Path
from argparse import ArgumentParser

ENTITY = "eleutherai"
PROJECTS = ["pythia", "pythia-extra-seeds"]


def get_runs_info(api: wandb.Api, entity: str, project: str) -> pd.DataFrame:
    runs = api.runs(f"{entity}/{project}")
    summary_list, config_list, name_list, id_list = [], [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items()})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

        # .id is the id of the run
        id_list.append(run.id)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "id": id_list,
        }
    )

    return runs_df


def filter_runs(df: pd.DataFrame) -> pd.DataFrame:
    r = df.loc[df["summary"].map(len) > 0].reset_index(drop=True).copy()

    to_extract = {
        "config": ("seed", "wandb_group"),
        "summary": ("_step", "_runtime"),
    }
    for name, values in to_extract.items():
        col = r[name].map(lambda ex: {k: v for k, v in ex.items() if k in values})
        r = pd.concat([r, pd.DataFrame(col.tolist())], axis=1)

    r = (
        r.query("(~_step.isna()) & (~_runtime.isna()) & (~wandb_group.str.contains('warmup'))")
        .assign(
            model_size=lambda _df: _df["wandb_group"].str.extract(r"-(.*)_", expand=False).str.split("_").str[0],
            runtime_min=lambda _df: _df["_runtime"] // 60,
        )
        .drop(columns=list(to_extract.keys()) + ["_runtime"])
        .rename(columns={"_step": "max_steps", "wandb_group": "group"})
        .convert_dtypes()
        .sort_values(["model_size", "seed"])
        .reset_index(drop=True)
    )

    return r


def download_data(api: wandb.Api, entity: str, project: str, run_id: str) -> pd.DataFrame:
    # get run path
    run = api.run(f"{entity}/{project}/{run_id}")

    # download data
    df = pd.DataFrame(run.scan_history(keys=["train/lm_loss", "_step"])).rename(
        columns={"_step": "step"}
    )

    return df



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--pythia_runs_path", type=str, default="./pythia_runs.tsv")
    parser.add_argument("--entity", type=str, default="eleutherai")
    parser.add_argument("--project", type=str, default="pythia-extra-seeds")
    parser.add_argument("--out_path", type=str, default="./raw_data")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    api = wandb.Api()

    runs_df = get_runs_info(api, args.entity, args.project)

    runs_df = filter_runs(runs_df)

    runs_df.to_csv(out_path / "runs.tsv", index=False, sep="\t")

    tasks = runs_df[["id", "model_size", "seed"]].to_dict("records")
    pbar = tqdm(total=len(tasks), desc="Downloading runs")
    for task in tasks:
        pbar.set_postfix_str(f"Run: {task['id']}")

        # check exists
        save_path = out_path / f"{task['model_size']}-seed{task['seed']}_{task['id']}.tsv"
        if save_path.exists():
            continue
        
        # download raw data
        df = download_data(api, args.entity, args.project, task["id"])
        
        # save
        df.to_csv(save_path, index=False, sep="\t")
        
        pbar.update(1)
