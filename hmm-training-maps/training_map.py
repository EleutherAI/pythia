from pandas import DataFrame
import pandas as pd
from typing import Optional, List, Dict
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import numpy as np
from hmmlearn import hmm
from sklearn.utils import check_random_state
import altair as alt
#import graphviz
#import pygraphviz as pgv
from collections import Counter
from graphviz import Digraph


def flatten(l):
    return [item for sublist in l for item in sublist]

def lengths2index(l):
        """Used for splitting predictions using np.array_split()
        Example:
            pred_lengths = self.lengths2index(self.lengths["train"])
            pred = self.hmm[NUM_STATES].predict(self.X["train"], self.lengths["train"])
            split_pred = np.array_split(pred, pred_lengths)
        """
        s_index = []
        prev = 0
        for s in l:
            s_index.append(s+prev)
            prev = s+prev
        return s_index[:-1]

class HMMTrainingMapData:
    """Prepares and stores data used for training the HMM training maps."""
    def __init__(self, data: DataFrame, random_state: int, metrics: List[str] = None, val_split: int = 0.2, pct_of_steps: float = 1.0):
        self._metrics = metrics
        self._data = data
        self.random_state = random_state
        self.val_split = val_split

        # Select the last time step
        if pct_of_steps == 1.0:
            self.last_step = data.step.max()
        else:
            self.last_step = data.loc[(data.step - pct_of_steps * data.step.max()).abs().idxmin()].step.item()
        
        self.X = {}
        self.lengths = {}
        self.timesteps = {}
        self._preprocess_data()

    def _preprocess_data(self):
        """Prepare data in format expected by hmmlearn.
        """        
        df = {}

        df["all"] = self.data.copy()
        
        df["all"] = df["all"][df["all"].step <= self.last_step]
        df["train"], df["val"] = train_test_split(df["all"], test_size=self.val_split, random_state=self.random_state)


        df["train"] = df["train"].sort_values(by=["seed","step"])
        df["val"] = df["val"].sort_values(by=["seed","step"])

        for split in ["all", "train", "val"]:
            self.X[split], self.timesteps[split], self.lengths[split] = [], [], []
            matrices = []
            for s in df["all"].seed.unique():
                df_ = df[split][df[split]["seed"]==s]
                matrices.append(df_[self.metrics])
                self.lengths[split].append(len(df_))
                self.timesteps[split].append(df_.step.to_list())
                
            self.X[split] = np.vstack(
                [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in matrices]
            )

    @property
    def metrics(self) -> List[str]:
        return self._metrics or self._data.set_index(["seed","step"]).columns.to_list()

    @property
    def data(self) -> DataFrame:
        return self._data
    
    @property
    def all(self):
        return {"X": self.X["all"], "lengths": self.lengths["all"], "timesteps": self.timesteps["all"]}

    @property
    def train(self):
        return {"X": self.X["train"], "lengths": self.lengths["train"], "timesteps": self.timesteps["train"]}

    @property
    def test(self):
        return {"X": self.X["test"], "lengths": self.lengths["test"], "timesteps": self.timesteps["test"]}

    @property
    def val(self):
        return {"X": self.X["val"], "lengths": self.lengths["val"], "timesteps": self.timesteps["val"]}

class HMMTrainingMap:
    """A trained HMM Training Map with n components for labeling training states.
    """

    def __init__(self, hmm, data: HMMTrainingMapData, state_colors: List[str] = None):
        self.hmm = hmm
        self.data = data
        self.state_colors = state_colors

        self.n_components = self.hmm.transmat_.shape[0]
        self._bag_of_states = None
    
    @property
    def plot_config(self):
        return {"state_colors": self.state_colors or ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']}

    @property
    def bag_of_states_distributions(self):
        """Distribution of states visited during training."""
        if self._bag_of_states is None:
            preds = self.hmm.predict(self.data.all["X"])
            index = lengths2index(self.data.all["lengths"])
            preds = np.array_split(preds, index)
            distributions = []
            for s in preds:
                total_states = len(s)
                c = Counter(s)
                bag_of_states = []
                for i in range(self.n_components):
                    bag_of_states.append(c[i] / total_states)
                distributions.append(bag_of_states)
            self._bag_of_states = np.array(distributions)
        return self._bag_of_states
    
    @property
    def detour_states(self):
        """A detour state is one which is not visited by all models, but only by some seeds during training.
        Returns distribution of the detour states (which may not be present)."""
        X = self.bag_of_states_distributions
        detour_states = []
        for state in range(X.shape[1]):
            if np.any(X[:, state] == 0):
                detour_states.append(state)
        assert detour_states, "There are no detour states!"
        return X[:,detour_states]

    def show_training_states(self, metric, data=None, scale="linear", time_pct=False, normalized=False, size_dot=100):
        """Returns a plot with each checkpoint labeled by the HMM."""
        seeds = flatten([[i]*l for i,l in enumerate(self.data.all["lengths"])])

        if data is not None:
            X = data[["seed","step"]+[metric]]
            X = X.sort_values(by=["seed","step"]).set_index(["seed","step"])
            X_ = pd.DataFrame(self.data.all["X"])
            X_["seed"] = seeds
            X_["step"] = flatten(self.data.all["timesteps"])
            X_ = X_.set_index(["seed","step"])
            X = X.join(X_)
            X = X.reset_index().dropna()

            timesteps = X.step.to_list()
            lengths_train = X.groupby('seed').count()["step"].to_list()
            X_train = X[X.columns.difference(["seed","step"]+[metric])].to_numpy()
            seeds = X.seed.to_list()
        else:
            X = self.data.data
            X_train = self.data.all["X"]
            lengths_train = self.data.all["lengths"]
            timesteps = flatten(self.data.all["timesteps"])

        if normalized:
            X = X.apply(zscore)
            
        X = X[metric].to_list()

        if time_pct:
            x_axis=alt.Axis(format='%', title="% of training")
        else:
            x_axis=alt.Axis(title="step")

        source = pd.DataFrame({"score": X, "state": self.hmm.predict(X_train, lengths_train), "step": timesteps, "seed": seeds})

        line = alt.Chart(source).mark_line().encode(
            x=alt.X('step:Q',scale=alt.Scale(type=scale), axis=x_axis),
            y=alt.Y('score:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title=f"{metric}{' (normalized)' if normalized else ''}")),
            #color="seed:N",
            #strokeDash="type:N",
        )

        dots = alt.Chart(source).mark_circle(size=size_dot).encode(
            x=alt.X('step:Q',scale=alt.Scale(type=scale), axis=x_axis),
            y=alt.Y('score:Q', scale=alt.Scale(zero=False)),
            color=alt.Color("state:N", scale=alt.Scale(range=self.plot_config["state_colors"])),
        )

        return (line+dots).facet(column="seed")
    
    def show(self, num_decimals=3, epsilon=0.001, fname="hmm_state_graph.png", model_seed=None):
        if model_seed is not None:
            preds = self.hmm.predict(self.data.X["all"])
            preds = self.select_for_seed(preds, model_seed, self.data.all["lengths"])
            components_for_seed = set(preds)
            state_transitions = self.select_state_transitions(preds)

        dot = Digraph(comment='Markov Chain')
        dot.attr(rankdir='LR', size='8,5')
        dot.attr('node', shape='circle')

        for n in range(self.n_components):
            color = self.plot_config["state_colors"][n]
            style = "filled"
            if model_seed is not None:
                if n not in components_for_seed:
                    color = "gray"
                    style = "dashed"
            dot.node(str(n), style=style, fillcolor=color)

        for i in range(self.n_components):
            for j in range(self.n_components):
                if self.hmm.transmat_[i][j] > epsilon:
                    color = "black"
                    linestyle = "solid"
                    if model_seed is not None:
                        if (i not in state_transitions) or (j not in state_transitions[i]):
                            color = "gray"
                            linestyle = "dotted"
                    dot.edge(str(i), str(j), label=str(round(self.hmm.transmat_[i][j], num_decimals)), _attributes = {"color":color, "linestyle":linestyle})
        return dot

    @staticmethod
    def select_state_transitions(preds):
        """Returns a list of state transitions present in preds"""
        i2j = {i: [] for i in list(set(preds))}
        for idx in range(len(preds)-1):
            i = preds[idx]
            j = preds[idx+1]
            i2j[i].append(j)
        return {i: set(j) for i,j in i2j.items()}

    @staticmethod
    def select_for_seed(preds, seed, lengths):
        """Only take predictions from preds for seed s"""
        index = lengths2index(lengths)
        return np.array_split(preds, index)[seed]

class HMMTrainingMapSelection:
    """Class for training and selecting HMM latent models for labeling training states.
    """

    def __init__(self, data: DataFrame, metrics: Optional[List] = None, val_split: float = 0.2, pct_of_steps: float = 1.0):
        """Initialization.

        Args:
            data (DataFrame): DataFrame where columns include "seed" and "step" and assumes all the other columns to be the metrics.
            metrics (Optional[List], optional): Column names in data that should be considered as metrics. Defaults to including all metrics.
            val_split (float, optional): Ratio of the data to use as the validation split. Defaults to 0.2.
            pct_of_steps (float, optional): Percentage of the steps to consider for training the HMMs. Defaults to 1.0.
        """        
        assert 0.0 < pct_of_steps <= 1.0
        
        self.data = HMMTrainingMapData(data, self.random_states["split"], metrics, val_split, pct_of_steps)

        # Save trained HMMs for n components
        self.hmm = {}

    @property
    def state_colors(self) -> List[str]: 
        #return ["#f49f74", "#aab8d8", "#eda0ce", "#76cbb4", "#b4dc66", "#fbe566"]
        # From https://gist.github.com/thriveth/8560036
        return ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    
    @property
    def random_states(self) -> Dict[str, int]:
        return {
            "split": 2546,
            "train": 5324,
            "val": [5329,2332,4642,6432,4754],
        }

    def _concat(self, l):
        """Concat lists of sequences and make sure it is not a singleton."""
        return np.concatenate(l, axis=1).reshape(-1, len(self.metrics))

    def model_selection(self, max_n_components: int = 8, only_n = None) -> DataFrame:
        """Run model selection procedure for n < max_n_components and k random states.

        Args:
            max_n_components (int, optional): Maximum number of components to consider. Defaults to 8.

        Returns:
            DataFrame: Results of model selection procedure.
        """        
        # h.monitor_.converged says true, while warning says not converged
        # Different metrics used for both convergence tests

        df = []

        if only_n:
            range_n = [only_n]
        else:
            range_n = range(1, max_n_components+1) 

        for n in range_n:
            best_score = -np.inf
            best_model = None

            for r_int in self.random_states["val"]:
                rs = check_random_state(r_int)
                h = hmm.GaussianHMM(
                    n_components=n, covariance_type="diag", n_iter=10000, random_state=rs
                )
                h.fit(self.data.train["X"], lengths=self.data.train["lengths"])
                
                try:
                    score = h.score(self.data.val["X"], lengths=self.data.val["lengths"])
                except ValueError:
                # Skip when ValueError: startprob_ must sum to 1 (got nan)
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = h

                df.append({
                    "n": n,
                    "rs": r_int,
                    "LL": score,
                    "BIC": h.bic(self.data.val["X"], lengths=self.data.val["lengths"]),
                    "AIC": h.aic(self.data.val["X"], lengths=self.data.val["lengths"]),
                    "converged": h.monitor_.converged
                })
            self.hmm[n] = (best_model, best_score)
        return pd.DataFrame(df)

    def show_model_selection(self, max_n_components=8, converged=False):
        """Run and show model selection plot for choosing number of components.
        For each number of components, will choose model with best score over k random seeds.

        Args:
            max_n_components (int, optional): Upper limit of number of components to consider. Defaults to 8.
            converged (bool, optional): Whether to only plot results for which hmmlearn states it has converged. Defaults to False.

        Returns:
            Altair plot.
        """        
        source = self.model_selection(max_n_components=max_n_components)

        source_ = source.copy()
        if converged:
            source_ = source_[source_.converged]
        source_ = source_[['AIC', 'BIC', 'LL', 'n', 'rs']]

        
        base = alt.Chart(source_).transform_fold(
                ['AIC', 'BIC'],
                as_=['metric', 'score']
            )
        
        line_l = base.mark_line().encode(
            x=alt.X('n:N',axis=alt.Axis(title="N")),
            y=alt.Y('mean(score):Q', axis=alt.Axis(title=f"Criterion Value (lower is better)"), scale=alt.Scale(zero=False)),
            color="metric:N",
            #strokeDash="type:N",
        )

        dots_l = base.mark_circle().encode(
            x=alt.X('n:N',axis=alt.Axis(title="N")),
            y=alt.Y('mean(score):Q', axis=alt.Axis(title=f"Criterion Value (lower is better)"), scale=alt.Scale(zero=False)),
            color="metric:N",
        )

        source_ = source.copy()[['LL', 'n', 'rs']]
        
        base = alt.Chart(source_).transform_fold(
                ['LL'],
                as_=['metric', 'score']
            )
        
        line_r = base.mark_line().encode(
            x=alt.X('n:N',axis=alt.Axis(title="N")),
            y=alt.Y('mean(score):Q', axis=alt.Axis(title=f"LL (higher is better)"), scale=alt.Scale(zero=False)),
            color="metric:N",
            #strokeDash="type:N",
        )

        dots_r = base.mark_circle().encode(
            x=alt.X('n:N',axis=alt.Axis(title="N")),
            y=alt.Y('mean(score):Q', axis=alt.Axis(title=f"LL (higher is better)"), scale=alt.Scale(zero=False)),
            color="metric:N",
        )

        band_r = base.mark_area(opacity=0.3).encode(
            alt.Y('max(score):Q', axis=alt.Axis(title=f"LL (higher is better)"), scale=alt.Scale(zero=False)),
            alt.Y2('min(score):Q', title=None),
            alt.X('n:N',axis=alt.Axis(title="N")),
        )

        return alt.layer((line_r+dots_r+band_r), (line_l+dots_l)).resolve_scale(
                    y='independent'
                ).properties(
                    width=400,
                    height=300
                )
    
    def get_training_map(self, n_components: int) -> HMMTrainingMap:
        """Get training map initialized with the best scoring HMM model for n_components.
        If no HMM has been found for n_components (e.g., during model selection procedure),
        will train a new one with model selection (i.e., choosing best from multiple random seeds).

        Args:
            n_components (int): Number of components to select for the HMM model.

        Returns:
            HMMTrainingMap: The training map with n_components components.
        """        
        if n_components not in self.hmm:
            self.model_selection(only_n=n_components)
            
            # np.random.seed(self.random_states["train"])
            # rs = check_random_state(self.random_states["train"])

            # model = hmm.GaussianHMM(
            #         n_components=n_components, covariance_type="diag", n_iter=10000, random_state=rs
            #     )
            # model.fit(self.data.train["X"], lengths=self.data.train["lengths"])
            # score = model.score(self.data.val["X"], lengths=self.data.val["lengths"])
            # self.hmm[n_components] = model, score
        return HMMTrainingMap(self.hmm[n_components][0], self.data, state_colors=self.state_colors)