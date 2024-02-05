import pandas as pd
import wandb
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from multiprocessing import Pool


ENTITY = "eleutherai"
PROJECTS = ["pythia", "pythia-extra-seeds"]


def download_data(run_id: str, model_size: str, seed: int, path: Path) -> None:
    out_path = path / f"{model_size}-seed{seed}.csv"
    if out_path.exists():
        return

    # get run path
    try:
        run = wandb.Api().run(f"{ENTITY}/{PROJECTS[0]}/{run_id}")
    except:
        run = wandb.Api().run(f"{ENTITY}/{PROJECTS[1]}/{run_id}")

    # download data
    df = (
        pd.DataFrame(run.scan_history(keys=["train/lm_loss", "_step"]))
        .rename(columns={"_step": "step"})
    )

    # save to disk
    df.to_csv(str(out_path), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pythia_runs_path", type=str, default="./pythia_runs.tsv")
    parser.add_argument("--out_path", type=str, default="./data")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    # runs = pd.read_csv(args.pythia_runs_path, sep="\t")
    # tasks = runs[["ID", "Model size", "Seed"]].to_dict("records")
    
    tasks = [{"ID": "bvgfairr", "Model size": "160m", "Seed": 4}]

    pbar = tqdm(total=len(tasks))

    def download_fn(d: dict) -> None:
        download_data(run_id=d["ID"], model_size=d["Model size"], seed=d["Seed"], path=out_path)

    with Pool(processes=8) as pool:
        for _ in pool.imap_unordered(download_fn, tasks):
            pbar.update(1)
