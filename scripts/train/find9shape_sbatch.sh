#!/bin/bash
# Submit this script from a FASRC login node with:
#   sbatch scripts/train/find9shape_sbatch.sh
#
# You can override Slurm resources at submit time, for example:
#   sbatch -p gpu_test --gpus=1 -t 0-01:00:00 scripts/train/find9shape_sbatch.sh
#   sbatch -p gpu_h200 --account=ydu_lab --gpus=4 -t 0-12:00:00 scripts/train/find9shape_sbatch.sh

#SBATCH -J dz_find9shape
#SBATCH -p gpu_h200
#SBATCH --account=ydu_lab
#SBATCH --gpus=4
#SBATCH -c 32
#SBATCH --mem=320G
#SBATCH -t 0-12:00:00
#SBATCH -o find9shape_%j.out
#SBATCH -e find9shape_%j.err

set -euo pipefail

cd /n/holylabs/ydu_lab/Lab/zhiyan/myProjects/dreamzero

source /n/home05/zhiyanli/miniforge3/etc/profile.d/conda.sh
conda activate dreamzero

export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
export FIND9SHAPE_DATA_ROOT=${FIND9SHAPE_DATA_ROOT:-/n/holylabs/ydu_lab/Lab/zhiyan/myProjects/data/find9shape_small}
export AGIBOT_CKPT_DIR=${AGIBOT_CKPT_DIR:-/n/home05/zhiyanli/checkpoints/DreamZero-AgiBot}
export OUTPUT_DIR=${OUTPUT_DIR:-/n/holylabs/ydu_lab/Lab/zhiyan/myProjects/dreamzero/checkpoints/dreamzero_find9shape_lora}

# Keep HuggingFace cache on the compute node's temporary storage when Slurm provides it.
export HF_HOME=${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf_cache_${USER:-zhiyanli}}

# Defaults are intentionally small for the first run. Override when submitting:
#   sbatch --export=ALL,MAX_STEPS=1000,SAVE_STEPS=200 scripts/train/find9shape_sbatch.sh
export MAX_STEPS=${MAX_STEPS:-100}
export SAVE_STEPS=${SAVE_STEPS:-100}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME: $HF_HOME"
echo "OUTPUT_DIR: $OUTPUT_DIR"

nvidia-smi

bash scripts/train/find9shape_training.sh
