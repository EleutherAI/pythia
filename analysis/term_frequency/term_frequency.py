import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

few_shot_list = [16, 8, 4, 2, 0]

eval_steps, max_steps = 13_000, 143_000
checkpoint_list = list(range(eval_steps, max_steps+eval_steps, eval_steps))

task = "num_reasoning_arithmetic_multiplication"

freq = Counter()
for checkpoint in tqdm(checkpoint_list):
    
    checkpoint_path = os.path.join("results", f"frequency_count_checkpoint_{checkpoint}.npy")
    _dict =  np.load(checkpoint_path, allow_pickle=True)[()]
    freq.update(_dict)

model_size = ["13b", "6.7b", "2.7b", "1.3b", "800m", "350m", "125m", "19m"]
x, y, z = [], [], []
for size in model_size:

    df = pd.read_csv(f"results/csv/pythia-{size}-deduped/term_frquency_all_shots.csv")
    df = df[df["task"].str.contains(task) & (df["checkpoint"] == 143000) & (df["fewshot"] == 16)]

    for i in range(0, 100):
        i = str(i)
        x.append(freq[i])
        y.append(df[df['task'].str.contains("_"+i)]['acc'].values[0])
        z.append(size)

    data = pd.DataFrame(
        data={
            'x': x,
            'y': y,
            'z': z,
            })

    ax = sns.regplot(
        data=data, x="x", y="y",# hue="z",
        fit_reg=True,
        # ci=None,
        label=size,
        logx=True,
        scatter=False
        )
    
ax.set(xscale="log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Avg. Accuracy")
# ax.set_title("Multiplication Performance, 13b-deduped @143000 16-shot")
ax.set_title("Multiplication Performance, All sizes @143000 16-shot")
plt.legend(fontsize=10)
plt.savefig('all_size-143000-arithmetic-freq-acc.png', dpi=200)
plt.clf()