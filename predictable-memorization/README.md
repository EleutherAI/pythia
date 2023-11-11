# [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/pdf/2304.11158.pdf)

This folder documents our work using Pythia to study memorization of particular sequences in the training dataset, and includes instructions to reproduce our analyses where possible.

## Reproducing Memorization Results
The memorization evaluation script `memorization/eval_memorization.py` assumes that you are running the script in a distributed process, ideally in slurm. It also assumes that you are using s3 to load and save Pythia's preshuffled Pile datasets (refer [here](https://github.com/EleutherAI/pythia/blob/main/README.md#dataset-viewer) for more details on how to download them), though using a local filesystem for the preshuffled datasets is also supported.

If you want to reproduce the evaluation, consider the following steps.

1. Change `prefix` local variable of `generate_function()` to point to the right document path.

2. If you are not using [Slurm](https://slurm.schedmd.com/documentation.html), You need to change global variables inside the script, like `RANK` and `NUM_PROCS` (world size) to point to the right environment variables.

3. Change `cache_dir` of model being loaded (line 172) to point to locally saved directory of the model. This is necessary as we **donot** want to load the same model multiple times. Doing so will lead to errors.

4. This script additionally saves results to aws s3 buckets (line 205). If you would like to save the results locally instead, you can do so by saving `memorization_evals` as a csv instead.

5. You should ideally be able to run this script now on slurm (see `memorization/multinode_runner.sbatch`) for an example sbatch script.

6. If you are using a different distributed client instead, you will need to pass `MODEL` and `CHECKPOINT` variables appropriately (see `memorization/multinode_runner.sbatch`) for an example 

7. These csvs can then be combined by simple pandas concatenation. See `memorization/eda.ipynb` for an example.

8. You can now generate plots too by following `memorization/eda.ipynb`.

## Reproducing Figures

Refer to `memorization/eda.ipynb` for details on replication.

## Reproducing Scaling Laws Plots

Refer to `memorization/eda.ipynb` for details on replication.

## Citation Details

If our work and data is useful to your research, please consider citing our paper via:

```
@inproceedings{biderman2023emergent,
      title={Emergent and Predictable Memorization in Large Language Models}, 
      author={Biderman, Stella and Prashanth, USVSN Sai and Sutawika, Lintang and Schoelkopf, Hailey and Anthony, Quentin and Purohit, Shivanshu and Raff, Edward},
      journal={Advances in Neural Information Processing Systems},
      year={2023}
}
```
