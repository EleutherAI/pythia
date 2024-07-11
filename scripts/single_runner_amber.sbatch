#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --hint=nomultithread        
#SBATCH --mem=40G
#SBATCH --output=../logs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=../logs/%j.out # Set this dir where you want slurm outs to go
#SBATCH --gres=gpu:a100:1
#SBATCH -p normal

TRAIN_PATH=/om/user/sunnyd/workspace/pythia/predictable_memorization/scripts
source activate pythia

export TRANSFORMERS_CACHE=/om/tmp/amber_cache
export HF_DATASETS_CACHE="/om/tmp/amber"
export DATA_CHECKPOINT=100
export CHECKPOINT=$1
export JOB_ID=$2
export NUM_BLOCKS=16

python $TRAIN_PATH/eval_memorization_amber.py

wait
