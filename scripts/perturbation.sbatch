#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --hint=nomultithread        
#SBATCH --mem=24G
#SBATCH --output=../logs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=../logs/%j.out # Set this dir where you want slurm outs to go
#SBATCH --gres=gpu:a100:1
#SBATCH -p normal

TRAIN_PATH=/om/user/sunnyd/workspace/pythia/predictable-memorization
source activate pythia

export WORKER_ID=$1
export SUFFIX=_control

TRANSFORMERS_CACHE=/om/user/sunnyd/transformers_cache python perturbation.py

wait
