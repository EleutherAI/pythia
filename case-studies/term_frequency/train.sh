#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=pile-t5x
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/fsx/lintangsutawika/improved_t5/logs/%x_%j.out
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --account=neox

module load openmpi
module show cuda/11.7
module load cuda/11.7

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
# export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
# export NCCL_SOCKET_IFNAME="eno1,eth0"
# export NCCL_BUFFSIZE=1048576

export NCCL_SOCKET_IFNAME=^docker0,lo

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

# export CUDA_LAUNCH_BLOCKING=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

source /fsx/lintangsutawika/01-project-pythia/bin/activate

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr:1234
echo "MASTER_ADDR="$MASTER_ADDR
# export ADDR="$(hostname -f):29500"


srun --account neox \
    python run_model_eval.py \
       --output_dir ../../results/ \
       --file_name_affix early_checkpoints_1000 \
       --few_shot_list 4 \
       --task_names trivia_qa \
       --checkpoint_list 1000 \
       --model_name EleutherAI/pythia-v1.1-12b-deduped
