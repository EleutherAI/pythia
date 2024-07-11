import click
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

import numpy as np

@click.command()
@click.option('--input_dir', type=str, default="../results/mem_once")
@click.option('--output', type=str, default="../results/mem_once")
def main(input_dir, output):
    ["index"] + [f"match{2**i}" for i in range(0, 9)]
    all_dfs = []

    for i in range(10000, 20001, 1000):
        df = pl.scan_csv(f"../results/memorization-dyn-count/evals-running/memorization_1b-v0_{i}_10240000_lev/*.csv", has_header=False, new_columns = ["index", "longest_match", "overlap", "lev"])
        df = df.with_columns(checkpoint = pl.lit(i))
        all_dfs.append(df)
    all_dfs = pl.concat(all_dfs)

    df = all_dfs.collect().to_pandas().set_index(["index", "checkpoint"]).sort_values(by=["index", "checkpoint"])
    df = df[~df.index.duplicated(keep='first')]
    df["diff"] = df["lev"].shift(-1) - df["lev"]
    indices = np.load(f'{input_dir}/indices.npy')
    complexity = pd.read_csv(f'{input_dir}/zcomplexity.csv', index_col=0)

    # summary = np.load(f"{input_dir}/repeat_summary.npz")
    counts = pd.read_csv(f"{input_dir}/repeat_count.csv", index_col=0)


    df = df.join(pd.pivot(counts, columns="size", values="count", index=["index", "checkpoint"]), how='inner')
    comp = df.apply(lambda x: complexity.loc[x.name[0], "complexity"], axis=1)
    df["complexity"] = comp
    df["cumsum30"] = df.groupby("index")[33].transform(pd.Series.cumsum)

    df.to_parquet(f'{output}/analysis.parquet.gzip', compression='gzip') 

if __name__ == "__main__":
    main()