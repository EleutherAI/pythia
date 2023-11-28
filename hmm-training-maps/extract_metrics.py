import torch
import numpy as np
import ray
from ray.experimental import tqdm_ray
import pandas as pd
from pathlib import Path
import logging
import psutil
from tqdm.auto import tqdm
import argparse
from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints
from typing import Dict
from concept_erasure import LeaceEraser

MAN_WORDS = ['man',
 'boy',
 'guy',
 'gentleman',
 'lord',
 'male',
 'masculine',
 'king',
 'prince',
 'monk',
 'wizard',
 'father',
 'dad',
 'brother',
 'nephew',
 'uncle',
 'grandfather',
 'son',
 'groom',
 'husband',
 'boyfriend',
 'he',
 'him',
 'himself',
 'his',
 'his',
 'men',
 'boys',
 'gentlemen',
 'lords',
 'males',
 'fathers',
 'brothers',
 'sons',
 'husbands',
 'Man',
 'Boy',
 'Lord',
 'Male',
 'King',
 'Prince',
 'Duke',
 'Wizard',
 'Father',
 'Dad',
 'Brother',
 'Uncle',
 'Husband',
 'He',
 'Him',
 'His',
 'Men',
 'Boys',
 'Lords',
 'Kings',
 'Brothers',
 'Sons']

WOMAN_WORDS = ['woman',
 'girl',
 'gal',
 'lady',
 'lady',
 'female',
 'feminine',
 'queen',
 'princess',
 'nun',
 'witch',
 'mother',
 'mum',
 'sister',
 'niece',
 'aunt',
 'grandmother',
 'daughter',
 'bride',
 'wife',
 'girlfriend',
 'she',
 'her',
 'herself',
 'her',
 'hers',
 'women',
 'girls',
 'ladies',
 'ladies',
 'females',
 'mothers',
 'sisters',
 'daughters',
 'wives',
 'Woman',
 'Girl',
 'Lady',
 'Female',
 'Queen',
 'Princess',
 'Duchess',
 'Witch',
 'Mother',
 'Mum',
 'Sister',
 'Aunt',
 'Wife',
 'She',
 'Her',
 'Her',
 'Women',
 'Girls',
 'Ladies',
 'Queens',
 'Sisters',
 'Ladies']

class SimpleGenderEraser:
    """Train a simple gender eraser for the input embeddings of a transformer."""

    # TODO: Currently, the word lists are duplicated across the different Ray cluster nodes. Ideally, these would be explicitly shared beforehand.

    def __init__(self, model, tokenizer):
        self.input_embeddings = model.get_input_embeddings()
        self.tokenizer = tokenizer
        self.X, self.Y = None, None

    @property
    def data(self):
        if not self.X and not self.Y:
            X, Y = [], []
            for w in MAN_WORDS:
                w = self.tokenizer.encode(w, return_tensors='pt')
                # Skip words that are tokenized as multiple tokens
                if w.shape[1] > 1:
                    continue
                x = self.input_embeddings(w).squeeze(0)
                X.append(x)
                Y.append(0)

            for w in WOMAN_WORDS:
                w = self.tokenizer.encode(w, return_tensors='pt')
                # Skip words that are tokenized as multiple tokens
                if w.shape[1] > 1:
                    continue
                x = self.input_embeddings(w).squeeze(0)
                X.append(x)
                Y.append(1)

            self.X = torch.concat(X)
            self.Y = torch.Tensor(Y)
        return self.X, self.Y

    def get_eraser(self):
        X, Y = self.data
        print(X.shape)
        print(Y.shape)
        return LeaceEraser.fit(X, Y)


def flatten_concat(list_of_X):
    return torch.concatenate([X_.flatten() for X_ in list_of_X]).flatten()


def get_matrix_metrics(X: torch.Tensor) -> Dict[str, float]:
    """Extracts metrics from matrix X as defined by Hu et al. (2023) used for training HMM latent state models.
    Original implementation from code Hu et al.

    Args:
        X (torch.Tensor): Tensor (e.g. weight or bias matrix).

    Returns:
        dict[str, float]: Dictionary mapping the metric names to the numeric results.
    """
    if torch.isnan(X).any():
        return
    if torch.isinf(X).any():
        return
    if torch.isneginf(X).any():
        return

    def get_flattened_l1_norm(x):
        return torch.linalg.vector_norm(x, ord=1)

    def get_flattened_l2_norm(x):
        return torch.linalg.vector_norm(x, ord=2)

    def get_spectral_norm(X):
        return torch.linalg.matrix_norm(X, ord=2)

    l1 = get_flattened_l1_norm(X).item()
    l2 = get_flattened_l2_norm(X).item()

    trace = torch.trace(X).item()
    spectral = get_spectral_norm(X).item()
    singular_vals = torch.svd(X, compute_uv=False).S
    singular_vals[singular_vals < 1e-5] = 0.0
    mean = torch.mean(singular_vals).item()
    var = torch.var(singular_vals).item()

    return {
        "L1": l1,
        "L2": l2,
        "trace": trace,
        "lambda_max": spectral,
        "L1/L2": l1 / l2,
        "trace/lambda": trace / spectral,
        "mu_lambda": mean,
        "sigma_lambda": var,
        # "singular_values": singular_vals.tolist(),
    }


class CheckpointMetricExtractor:
    """Class for extracting Hu et al. (2023) metrics for a model checkpoint."""

    def __init__(self, model, config, tokenizer=None, eraser=False):
        self.metrics = {"step": config["step"], "seed": config["seed"]}
        self.checkpoint = model
        self.tokenizer = tokenizer
        if eraser:
            assert tokenizer is not None
        self.eraser = eraser

    @torch.no_grad()
    def _prepare_weights_biases(self):
        """Collect all weight and bias matrices for this model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weights and biases matrices.
        """
        # Collect all the weight and bias matrices
        weights = []
        biases = []

        for name, param in self.checkpoint.named_parameters():
            # SKIP = ["LayerNorm", "layer_norm", "layernorm"]
            # embed_in, embeddings
            if name.startswith("embed"):
                continue
            if "norm" in name.lower():
                continue
            if name.endswith(".weight"):
                weight = param.data.view(param.shape[0], -1)
                weights.append(weight)
            elif name.endswith(".bias"):
                bias = param.data.view(param.shape[0], -1)
                biases.append(bias)
        return weights, biases
    
    @torch.no_grad()
    def _prepare_weights_biases_eraser(self):
        """Collect all weight and bias matrices for an eraser model trained on the model's hidden states.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weights and biases matrices.
        """
        eraser = SimpleGenderEraser(self.checkpoint, self.tokenizer).get_eraser()

        # Collect all the weight and bias matrices
        weights = []
        biases = []

        weights.append(eraser.proj_left)
        weights.append(eraser.proj_right)
        biases.append(eraser.bias)
        return weights, biases

    def compute_metrics(self):
        """
        https://arxiv.org/pdf/2308.09543.pdf Table 3
        """
        if self.eraser:
            weights, biases = self._prepare_weights_biases_eraser()
        else:
            weights, biases = self._prepare_weights_biases()

        for w in weights:
            metrics_ = get_matrix_metrics(w)

            for key, val in metrics_.items():
                self.metrics.setdefault(key, []).append(val)

        for key in metrics_.keys():
            self.metrics[key] = np.mean(self.metrics[key])

        for w, l in [(weights, "w"), (biases, "b")]:
            flattened_w = flatten_concat(w)
            self.metrics["mu_" + l] = torch.mean(flattened_w).item()
            self.metrics["sigma_" + l] = torch.var(flattened_w).item()
            self.metrics["median_" + l] = torch.median(flattened_w).item()

        return self.metrics


@ray.remote
class RayMetricExtractor:
    """Helper class for running parallel metric extraction with ray."""

    def __init__(self, checkpoints, eraser=False):
        self.checkpoints = checkpoints
        self.eraser = eraser

    def compute_metrics(self, tqdm_pos=0):
        results = []
        for ckpt in tqdm_ray.tqdm(
            self.checkpoints, total=len(self.checkpoints), position=tqdm_pos
        ):
            me = CheckpointMetricExtractor(ckpt.model, ckpt.config, tokenizer=ckpt.tokenizer, eraser=self.eraser)
            result = me.compute_metrics()
            results.append(result)
        return results


class MetricExtractor:
    """Class for managing the metric extraction for a set of checkpoints."""

    def __init__(self, checkpoints, results_fp=None, eraser=False):
        self.checkpoints = checkpoints
        self.results = None
        if results_fp:
            self.results = Path(results_fp)
        self.eraser = eraser

    @property
    def metrics(self):
        return [
            "L1",
            "L2",
            "L1/L2",
            "mu_w",
            "median_w",
            "sigma_w",
            "mu_b",
            "median_b",
            "sigma_b",
            "trace",
            "lambda_max",
            "trace/lambda",
            "mu_lambda",
            "sigma_lambda",
        ]

    def _clean_df(self, df):
        df.sort_values(by=["seed", "step"], inplace=True)
        df = df[["seed", "step"] + self.metrics]
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        return df

    def get_metrics(self, rerun=False, parallel=1):
        if self.results:
            if self.results.exists() and not rerun:
                return self._clean_df(pd.read_csv(self.results, sep="\t"))

        if parallel > 1:
            # Running metric extraction for checkpoints in parallel using ray.
            num_cpu = psutil.cpu_count(logical=False)

            if ray.is_initialized():
                ray.shutdown()
            ray.init(num_cpus=num_cpu, logging_level=logging.ERROR)

            num_actors = min(parallel or num_cpu, len(self.checkpoints))
            logging.debug(f"{num_actors} max total actors")

            actors = []

            for chunk in self.checkpoints.split(num_actors):
                actors.append(RayMetricExtractor.remote(chunk, eraser=self.eraser))

            metrics = ray.get(
                [c.compute_metrics.remote(tqdm_pos=i) for i, c in enumerate(actors)]
            )
            metrics = [item for sublist in metrics for item in sublist]
            ray.shutdown()

        else:
            # Running metric extraction sequentially.
            metrics = []
            for ckpt in tqdm(self.checkpoints):
                me = CheckpointMetricExtractor(ckpt.model, ckpt.config, eraser=self.eraser, tokenizer=ckpt.tokenizer)
                result = me.compute_metrics()
                metrics.append(result)

        # Creating a dataframe with the metrics
        df = pd.DataFrame(metrics)
        df = self._clean_df(df)

        if self.results:
            df.to_csv(self.results, sep="\t", index=False)
        return df


if __name__ == "__main__":
    """Example usages:
    >> python extract_metrics.py pythia160m --results_fp "results/Pythia160m_Hu_metrics.tsv" --print
    >> python extract_metrics.py bert --results_fp "results/Pythia160m_Hu_metrics.tsv" --print

    Extract metrics from gender probe instead of the model hidden states:
    >> python extract_metrics.py pythia160m --results_fp "results/Pythia160m_input_embeddings_gender_LEACE_metrics.tsv" --eraser --print
    """
    parser = argparse.ArgumentParser(
        description="Extract metrics for model checkpoints."
    )
    parser.add_argument("model", type=str, help="bert or e.g. pythia160m.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel processes using Ray.",
    )
    parser.add_argument(
        "--results_fp",
        type=str,
        required=True,
        help="Number of parallel processes using Ray.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Whether to rerun if results file (results_fp) already exists.",
    )
    parser.add_argument(
        "--eraser",
        action="store_true",
        help="Whether to extract metrics from LEACE eraser.",
    )
    parser.add_argument(
        "--print", action="store_true", help="Whether to print the results."
    )

    args = parser.parse_args()

    if args.model == "bert":
        checkpoints = MultiBERTCheckpoints()
    elif args.model.startswith("pythia"):
        model_size = int(args.model.strip("pythia").strip("m"))
        checkpoints = PythiaCheckpoints(size=model_size)
    me = MetricExtractor(checkpoints, results_fp=args.results_fp, eraser=args.eraser)
    df_metrics = me.get_metrics(parallel=args.parallel, rerun=args.rerun)

    if args.print:
        print(df_metrics.to_markdown())
