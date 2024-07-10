# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

import numpy as np

# %% [markdown]
# # Process Model output

# %%

["index"] + [f"match{2**i}" for i in range(0, 9)]
all_dfs = []
for i in range(10000, 20001, 1000):
    df = pl.scan_csv(f"results/memorization-dyn-count/evals-running/memorization_1b-v0_{i}_10240000_lev/*.csv", has_header=False, 
            new_columns = ["index", "longest_match", "overlap", "lev"])
    # df["index"] = df["index"].cast(pl.Int64)
    df = df.with_columns(checkpoint = pl.lit(i))
    all_dfs.append(df)
all_dfs = pl.concat(all_dfs)

# %%
d = all_dfs.collect().pivot("lev", "index", "checkpoint", aggregate_function="mean")

f, ax = plt.subplots(1, 9, figsize=(50, 6))
for i in np.arange(1, 10):
    ax[i-1].hist(d[str(i*1000+11000)] - d[str(i*1000 + 10000)], bins=50)
    ax[i-1].set(yscale='log')
    ax[i-1].set_xlabel(f'ckpt{i*1000 + 1000} (mem) - ckpt{i*1000}  (mem)')
ax[0].set_ylabel('counts')


# %%
f, ax = plt.subplots(figsize=(10, 100))

df_t = d.to_pandas()
df_t = df_t[(df_t.iloc[:, 1:].max(axis=1)) > 32]
sns.heatmap(df_t.iloc[:, 1:], ax=ax)


# %% [markdown]
# # Repeats

# %%
df = all_dfs.collect().to_pandas().set_index(["index", "checkpoint"]).sort_values(by=["index", "checkpoint"])
df = df[~df.index.duplicated(keep='first')]
df["diff"] = df["lev"].shift(-1) - df["lev"]
df

# %%
complexity = np.load('zcomplexity.npy')
ser = pd.DataFrame(complexity)
ser.index = np.arange(len(complexity)) * 20 + 1024 * 10000
ser.columns = [30] #np.arange(1, 256)
# ser[0] = 1.0
plt.hist(ser.loc[:, 30])
plt.title("Complexity (unique tokens)/total")

# %%
topk = np.load("topk.npy")
repeat_counts = []
for i in range(10):
    repeat_counts.append(np.load(f"repeat_count_{i}.npy"))
counts = np.stack(repeat_counts, axis=1)

index, checkpoint, size = np.unravel_index(np.arange(len(counts.reshape(-1))), counts.shape)
cts = pd.DataFrame(
    {"counts": counts.reshape(-1), "index": index * 20 + 10000 * 1024, "checkpoint": checkpoint*1000 + 10000, "size": size*10})
df = df.join(
    pd.pivot(cts, columns="size", values="counts", index=["index", "checkpoint"]), how='inner')
comp = df.apply(lambda x: ser.loc[x.name[0], 30], axis=1)
df["complexity"] = comp
df["cumsum30"] = df.groupby("index")[30].transform(pd.Series.cumsum)

# %%
df.to_parquet('analysis.parquet.gzip',
              compression='gzip') 

# %% [markdown]
# # Visualize

# %%
df = pd.read_parquet('analysis.parquet.gzip')  

# %%
df

# %%
plt.hist2d(df[df.index.get_level_values(1) == 10000]["lev"],
df[df.index.get_level_values(1) == 11000]["lev"], bins=40, norm='log')

# %%
df_t = df.groupby(pd.cut(np.log(df["cumsum30"]), 100)).mean()["lev"]
df_t
bins = df_t.index.map(lambda x: x.right).to_numpy().astype(float)
plt.plot(np.exp(bins), df_t)
plt.xscale('log')
plt.xlim(5, 1e5)
plt.xlabel("cumulative repeats (size 30)")
plt.ylabel("Minimum edit distance (Levenshtein distance)")

# %%
df_t = df.groupby([pd.cut(df["complexity"], 50), pd.cut(np.log(df["cumsum30"]), 50)]).mean()
sns.heatmap(df_t["lev"].reset_index().pivot(columns="cumsum30", index="complexity", values="lev"))
plt.figure()
df_t = df.groupby([pd.cut(df["complexity"], 50), pd.cut(np.log(df["cumsum30"]), 50)]).var()
sns.heatmap(df_t["lev"].reset_index().pivot(columns="cumsum30", index="complexity", values="lev"))
# bins = df_t.index.map(lambda x: x.right).to_numpy().astype(float)
# plt.plot(bins, df_t)
# plt.xlim(0, 1)
# plt.xlabel("Complexity")
# plt.ylabel("Minimum edit distance (Levenshtein distance)")

# %%
df_t = df.groupby(pd.cut(df["complexity"], 100)).mean()["lev"]

# %%
df

# %%
df_t = df.loc[df["complexity"] >0.0]
plt.hist2d(np.log(df_t["cumsum30"].astype(float)+1), np.clip(df_t["lev"], 0, 60), 
#            xscale='log',
#            bins=(50, 50),
#            range=[[0, 100], [0, 50]],
           bins=[np.linspace(0, np.log(1000), 51), np.linspace(0, 60, 30)], 
           norm='log',
#             alpha=0.1
          );
plt.xlabel("log(repeats)")
plt.ylabel("longest match")
plt.title("match-repeat (complexity > 0.8)")

# %%
f, ax = plt.subplots(2, 5, figsize=(30, 12))
for i in range(10):
    comp = 0.025 * i + 0.4
    df_t = df.loc[(df["complexity"] > comp) & (df["complexity"] <= comp+0.05)]
    ax[i // 5, i % 5].hist2d(np.log(df_t["cumsum30"].astype(float)+1), np.clip(df_t["lev"], 0, 60), 
    #            xscale='log',
    #            bins=(50, 50),
    #            range=[[0, 100], [0, 50]],
               bins=[np.linspace(0, np.log(1000), 51), np.linspace(0, 60, 30)], 
               norm='log',
    #             alpha=0.1
              );
    ax[i // 5, i % 5].set(xlabel="log(repeats)", ylabel="longest match", title=f"match-repeat ({comp} < complexity < {comp+0.05:0.2f})")

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(np.log(df["cumsum30"].astype(float)+1), df["complexity"], np.clip(df["longest_match"], 0, 100))

# %%
df.index.get_level_values(1)

# %%
df_t["10"]

# %%
f, ax = plt.subplots(2, 5, figsize=(30, 12))
for i in range(6):
    comp = 0.025 * i + 0.4
    df_t = df.loc[(df["complexity"] > comp) & (df["complexity"] <= comp+0.05) & (df.index.get_level_values(1) > 3000)]
    ax[i // 5, i % 5].hist2d(
        np.log(df_t["10"].astype(float)+1), 
        df_t["diff"], 
    #            xscale='log',
    #            bins=(50, 50),
    #            range=[[0, 100], [0, 50]],
               bins=[np.linspace(0, np.log(1000), 51), np.linspace(-60, 60, 30)], 
               norm='log',
    #             alpha=0.1
              );
    ax[i // 5, i % 5].set(xlabel="log(repeats) size 10", ylabel="longest match", title=f"match-repeat ({comp} < complexity < {comp+0.05})")

# %%
f, ax = plt.subplots(2, 5, figsize=(30, 12))
for i in range(6):
    comp = 0.025 * i + 0.4
    df_t = df.loc[(df["complexity"] > comp) & (df["complexity"] <= comp+0.05)]
    ax[i // 5, i % 5].hist2d(
        df_t["lev"],
        df_t["diff"], 
    #            xscale='log',
    #            bins=(50, 50),
    #            range=[[0, 100], [0, 50]],
               bins=[np.linspace(0, 60, 51), np.linspace(-60, 60, 30)], 
               norm='log',
    #             alpha=0.1
              );
    ax[i // 5, i % 5].set(xlabel="lev distance", ylabel="change in edit distance", title=f"match-repeat ({comp} < complexity < {comp+0.05})")

# %%
df.groupby(df.index.get_level_values(0)).agg("mean")["lev"]

# %%
x = df.groupby(df.index.get_level_values(0)).agg("mean")["lev"]
y = df.groupby(df.index.get_level_values(0)).agg("var")["lev"]
plt.hist2d(x, y, norm='log', bins=50)

x = np.linspace(0, 64, 100)
y = 64 * x - x ** 2
plt.plot(x, y)

# %%
np.minimum(x, np.abs(x - 60))

# %%
f, ax = plt.subplots(figsize=(10, 100))

df_t = df["longest_match"].unstack(1).loc[(df["longest_match"].unstack(1) > 32).max(axis=1)]
df_t
sns.heatmap(df_t, ax=ax)

# %%
from sklearn.decomposition import PCA
pca = PCA(6)
dim = pca.fit_transform(df_t.dropna().to_numpy())

# %%
f, ax = plt.subplots()
sns.heatmap(pca.components_)
ax.set(ylabel='component number', xlabel='time')

# %%
pca.explained_variance_ratio_

# %%
f, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(np.cumsum(pca.explained_variance_ratio_))
ax[0].set(title='explained variance')
ax[1].scatter(dim[:, 0], dim[:, 1])
ax[1].set(title='First 2 pcs')

# %%
(df["longest_match"].unstack(1) > 32).sum(axis=1) 

# %%
# Analyze hard matches

# %%
hard_matches = df1[(df1["complexity"]>0.6) & (df1["longest_match"] > 32)] #& (df1[10] < 10)]
hard_matches.index.get_level_values(0).unique()

# %%
plt.hist2d(df1["cumsum30"], df1["longest_match"], 
           bins=(np.linspace(0, 100, 20), np.linspace(0, 250, 30)),
          norm='log');
# plt.ylim(0, 30)
# plt.xlim(0, 50)

# %%


# %%
df2 = pd.melt(df1.reset_index(), id_vars=["index", "checkpoint", "longest_match"], value_vars = np.arange(0, 26) * 10)
df2

# %%
# g = sns.FacetGrid(df1[df1["variable"] % 40 == 0], col="checkpoint", row="value")
# sns.relplot(df1, x="value", y="diff")

# %%
# f, ax = plt.subplots(13, 10, figsize=(120, 300))
# for x, i in enumerate(np.arange(1000, 10001, 1000)):
#     for y, j in enumerate(np.arange(10, 261, 20)):
#         df_t = df1[(df1["variable"] == j) & (df1["checkpoint"] == i)]
# #         ax[y, x].scatter(df_t["value"], df_t["diff"])
#         ax[y, x].hist2d(df_t["value"], df_t["diff"], bins=50, norm='log')
#         print(i, j)

# %%
df_t = df2[(df2["variable"] == 30)]
plt.hist2d(df_t["value"], df_t["longest_match"], bins=[np.linspace(0, 50, 50), np.linspace(0, 50, 50)], norm='log')

# %% [markdown]
# # Individual runs

# %%
idx = 1060
# idx = 20

# %%
plt.plot(dyn.to_numpy()[0, 1:].T)
plt.ylim(0, 256)

# %%

df1[(df1["index"] == idx)]["longest_match"]

# %%
sns.relplot(df1[df1["index"] == idx], x='checkpoint', y='value', hue='variable')

# %% [markdown]
# # Regression Analysis

# %%
df_f = df.filter(pl.col("index") < 100000).collect()
df_f = df_f.to_pandas().set_index("index")

df_f["longest_match"] = 0
for i in range(9):
    df_f.loc[df_f[f"match{2**i}"] == 1.0, "longest_match"] = 2**i

df_f = df_f.groupby("index").agg(["mean", "std"])
df_f.columns = ['_'.join(col).strip() for col in df_f.columns.values]
df_f

# %%
topk = np.load("topk.npy")
repeat_count = np.load("repeat_count.npy")

# %%
checkpoint, index, size = np.unravel_index(np.arange(len(repeat_count.reshape(-1))), repeat_count.shape)
df_r = pd.DataFrame()
df_r["count"] = repeat_count.reshape(-1)
df_r["repeat_size"] = [f"repeat_{2**s}" for s in size] 
df_r["index"] = index
df_r["checkpoint"] = checkpoint * 1000 + 1000
df_r = df_r.pivot(index=["checkpoint", "index"], values="count", columns="repeat_size")
# df_r = df_f.set_index("index")
df_r = df_r.reset_index(level=[0])
df_r = df_r.groupby("index").agg("sum")
# df_r.columns = ['_'.join(col).strip() for col in df_r.columns.values]
df_r

# %%
data = topk[:, :, 1].T
index, checkpoint = np.unravel_index(np.arange(len(data.reshape(-1))), data.shape)
df_x = pd.DataFrame()
df_x["topmatch"] = data.reshape(-1)
df_x["checkpoint"] = checkpoint * 1000 + 1000
df_x["index"] = index
df_x = df_x.set_index("index")
df_x

# %%
# df_c = df_x.join(df_f, on=["index", "checkpoint"], how='inner')
# df_c = pd.merge(df_x, df_f,  how='inner', left_on=["index", "checkpoint"], right_on = ["index", "checkpoint"])
# df_c = pd.merge(df_c, df_r,  how='inner', left_on=["index", "checkpoint"], right_on = ["index", "checkpoint"])
df_c = pd.merge(df_r, df_f,  how='inner', left_on=["index"], right_on = ["index"])
df_c

# %%
sns.histplot(df_c, x="repeat_8", y="longest_match_std", bins=60)

# %%
sns.histplot(df_c, x="repeat_64", y="longest_match_mean", bins=80)

# %%
sns.relplot(df_c, x="longest_match_mean", y="longest_match_std")

# %%
# sns.relplot(df_c, x='repeat_8', y='match8')
f, ax = plt.subplots(8, 7, figsize=(80, 80))
for i in range(8): # match_num
    for j in range(7): # repeat
        match_num = 2**i
        repeat_num = 2**j
        ax[i, j].hist([df_c[df_c[f"match{match_num}"] == 0][f"repeat_{repeat_num}"], 
                       df_c[df_c[f"match{match_num}"] == 1][f"repeat_{repeat_num}"]], 
                      bins=np.linspace(0, 600, 20), density=True);
        ax[i, j].set(yscale='log')

# %%
# sns.relplot(df_c, x='repeat_8', y='match8')
f, ax = plt.subplots(8, 7, figsize=(80, 80))
for i in range(8): # match_num
    for j in range(7): # repeat
        match_num = 2**i
        repeat_num = 2**j
        ax[i, j].hist([df_c[df_c[f"match{match_num}"] == 0][f"repeat_{repeat_num}"], 
                       df_c[df_c[f"match{match_num}"] == 1][f"repeat_{repeat_num}"]], 
                      bins=np.linspace(0, 600, 20), density=True);
        ax[i, j].set(yscale='log')

# %%
f, ax = plt.subplots(1, 7, figsize=(80, 10))
for j in range(7):
    repeat_num = 2**j
    ax[j].hist(df_c[f"repeat_{repeat_num}"],
                      bins=np.linspace(0, 600, 20));
    ax[j].set(yscale='log')

# %%
sns.displot(df_c, x="")

# %%
X =df_c[["topmatch", "checkpoint"]].to_numpy()
y = df_c[[f"match{int(2**i)}" for i in range(0, 9)]]
y = df_c["match256"]

# %%
# plt.hist(X[y==1, 0])

plt.hist(X[y==1, 0], bins=np.linspace(0, 100, 100), density=True, alpha=0.4);
plt.hist(X[y==0, 0], bins=np.linspace(0, 100, 100), density=True, alpha=0.4);


# %%

plt.hist(X[y==1, 1], density=True, alpha=0.4);
plt.hist(X[y==0, 1], density=True, alpha=0.4);

# %%
log = LogisticRegression()
log.fit(X, y)

# %% [markdown]
# ## TODO
# - Compare the variance, slope, mean, etc. of the dynamics with number of repeats
# - Run the analysis for the last 100k sequences before end of the checkpoint
# - Run larger models (1b) 


