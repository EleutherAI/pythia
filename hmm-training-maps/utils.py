import json
import pandas as pd
import statsmodels.api as sm

SCORES = ["acc", "pct_stereotype", "likelihood_diff"]
STD_SCORES = ["acc_stderr", "pct_stereotype_stderr", "likelihood_diff_stderr"]

def load_results(fp_template):
    df = []
    for seed in [0,1,2,3,4]:
        with open(fp_template.format(seed), 'r') as f:
            r = json.loads(f.read())["results"]
            for task in r.keys():
                for v in r[task].keys():
                    if v.removesuffix(",none") in SCORES:
                        df.append({"seed": seed, "task": task, "metric": v.removesuffix(",none"), "score": r[task][v]})

    return pd.DataFrame(df).pivot_table(index="seed", columns=["task", "metric"], values="score")

def load_loss(fp_template, last_steps=True):
    df_train_loss = []
    for seed in [0,1,2,3,4]:
        df_ = pd.read_csv(fp_template.format(seed)).set_index("Step")
        df_["train loss"] = df_.mean(axis=1)
        df_["seed"] = seed
        df_ = df_.reset_index()
        df_train_loss.append(df_[["seed", "Step", "train loss"]])
    df_train_loss = pd.concat(df_train_loss)
    df_train_loss = df_train_loss.rename(columns={"Step": "step"})
    if last_steps:
        return df_train_loss[df_train_loss["step"]>142900].groupby("seed").mean().drop(columns="step")
    else:
        return df_train_loss

def regression_table(X, Y: pd.DataFrame, color: bool = True):
    df_corr = []

    for task in Y.columns:
        y = Y[task].to_numpy()
        mod = sm.OLS(y, X)
        res = mod.fit()
        if isinstance(task, tuple):
            df_corr.append( {"task": task[0], "metric": task[1], "R^2": res.rsquared, "F-statistic": res.fvalue, "F-statistic P-value": res.f_pvalue} )
        else:
            df_corr.append( {"task": task, "R^2": res.rsquared, "F-statistic": res.fvalue, "F-statistic P-value": res.f_pvalue} )

    df = pd.DataFrame(df_corr)
    if color:
        cmap = 'RdYlGn'
        df = df.style.format(precision=2).background_gradient(cmap=cmap, vmin=-1, vmax=1, subset=["R^2"])
    return df

def label_task2group(row):
   if row['task'].startswith("crows_pairs"):
      return 'gender'
   if row['task'].startswith("blimp"):
      return 'gender'
   if row['task'].startswith("train loss"):
       return 'loss'
   return 'other'


def combine_regression_tables(list_of_dfs, list_of_names, color=True, val="R^2"):
    dfs = []
    for df, name in zip(list_of_dfs, list_of_names):
        df["group"] = df.apply(label_task2group, axis=1)
        if "metric" in df.columns:
            df = df.set_index(["group", "task", "metric"])
        else:
            df  = df.set_index(["group", "task"])
        df["name"] = name
        dfs.append(df[["name",val]])
    df = pd.concat(dfs).pivot(columns="name", values=val)
    if color:
        cmap = 'RdYlGn'
        df = df.style.format(precision=2).background_gradient(cmap=cmap, vmin=-1, vmax=1)
    return df