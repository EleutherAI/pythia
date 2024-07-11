#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --hint=nomultithread        
#SBATCH --mem=24G
#SBATCH --output=../logs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=../logs/%j.out # Set this dir where you want slurm outs to go
#SBATCH --gres=gpu:a100:1
#SBATCH -p fiete

TRAIN_PATH=/om/user/sunnyd/workspace/pythia/predictable-memorization
source activate pythia

export IDX_DIR="../results/indices/0_0.npy"
export OUTPUT="_future"

TRANSFORMERS_CACHE=/om/user/sunnyd/transformers_cache python grad_steps.py

wait
