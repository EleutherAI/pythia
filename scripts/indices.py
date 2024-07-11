
import polars as pl
import numpy as np
import click


@click.command()
@click.option('--source', type=str, help='A string argument.', default="../results/memorization-dyn-count/evals-running/memorization_1b-v0")
@click.option('--dest', type=str, help='A string argument.', default="../results/indices")
def main(source, dest):
    ["index"] + [f"match{2**i}" for i in range(0, 9)]
    all_dfs = []

    for i in range(10000, 20001, 1000):
        df = pl.scan_csv(f"../results/memorization-dyn-count/evals-running/memorization_1b-v0_{i}_10240000_lev/*.csv", has_header=False, new_columns = ["index", "longest_match", "overlap", "lev"])
        df = df.with_columns(checkpoint = pl.lit(i))
        all_dfs.append(df)
    all_dfs = pl.concat(all_dfs)
    df = all_dfs.collect().to_pandas()
    # df.groupby("index").min().query("lev < 30").index
    np.save("../results/indices/mem_once", 
            np.array(df.groupby("index").min().query("lev < 30").index.tolist()))
    # .to_csv("../results/indices/mem_once.csv", index=False)

if __name__ == '__main__':
    main()