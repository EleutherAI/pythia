#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -p fiete
#SBATCH --mem=40G
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=../logs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=../logs/%j.out # Set this dir where you want slurm outs to go

source activate pythia
python compute_repeats.py