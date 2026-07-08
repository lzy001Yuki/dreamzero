#!/bin/bash
# DreamZero find9shape_small LoRA fine-tuning script.
#
# Usage:
#   bash scripts/train/find9shape_training.sh
#
# Common overrides:
#   NUM_GPUS=1 MAX_STEPS=1000 OUTPUT_DIR=/path/to/output \
#     bash scripts/train/find9shape_training.sh

set -e

export HYDRA_FULL_ERROR=1

REMOTE_PROJECT_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero"
REMOTE_DATASET_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/dreamzero/datasets"
REMOTE_CHECKPOINT_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero/checkpoints"
VENV_DIR=${VENV_DIR:-"$REMOTE_PROJECT_ROOT/.venv"}

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Prefer the repository you launch from over any editable/install copy in the venv.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Dataset path in LeRobot/GEAR format.
FIND9SHAPE_DATA_ROOT=${FIND9SHAPE_DATA_ROOT:-"$REMOTE_DATASET_ROOT/find9shape_small"}

# Output directory for training checkpoints.
OUTPUT_DIR=${OUTPUT_DIR:-"$REMOTE_CHECKPOINT_ROOT/dreamzero_find9shape_lora"}

# Default to visible GPUs; fall back to 1 when nvidia-smi is unavailable.
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi

# Model weight paths. Defaults point at the remote server checkpoint root.
# Override these variables if your checkpoint subdirectory names differ.
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$REMOTE_CHECKPOINT_ROOT/Wan2.1-I2V-14B-480P"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$REMOTE_CHECKPOINT_ROOT/umt5-xxl"}
AGIBOT_CKPT_DIR=${AGIBOT_CKPT_DIR:-"$REMOTE_CHECKPOINT_ROOT/DreamZero-AgiBot"}

# Keep Hub cache out of the project directory by default. This is still a local
# cache, but can live on node-local Slurm tmp or /tmp instead of project storage.
HF_CACHE_ROOT=${SLURM_TMPDIR:-/tmp}
export HF_HOME=${HF_HOME:-"$HF_CACHE_ROOT/hf_cache_${USER:-zhiyanli}"}

# Small-data defaults. Override these for a longer run.
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}

if [ ! -d "$FIND9SHAPE_DATA_ROOT" ]; then
    echo "ERROR: find9shape dataset not found at $FIND9SHAPE_DATA_ROOT"
    echo "Set FIND9SHAPE_DATA_ROOT to your LeRobot-format find9shape dataset."
    exit 1
fi

if [ ! -d "$AGIBOT_CKPT_DIR" ]; then
    echo "ERROR: DreamZero-AgiBot checkpoint not found at $AGIBOT_CKPT_DIR"
    echo "Set AGIBOT_CKPT_DIR to your DreamZero-AgiBot checkpoint directory."
    exit 1
fi

if [ -n "$WAN_CKPT_DIR" ]; then
    WAN_ARGS=(
        "dit_version=$WAN_CKPT_DIR"
        "text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
        "image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        "vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    )
else
    WAN_ARGS=(
        "dit_version=null"
        "text_encoder_pretrained_path=null"
        "image_encoder_pretrained_path=null"
        "vae_pretrained_path=null"
    )
fi

torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/find9shape_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=17 \
    action_horizon=8 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=4 \
    num_action_per_block=8 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate="$LEARNING_RATE" \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps="$SAVE_STEPS" \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    find9shape_data_root="$FIND9SHAPE_DATA_ROOT" \
    "${WAN_ARGS[@]}" \
    tokenizer_path="$TOKENIZER_PATH" \
    pretrained_model_path="$AGIBOT_CKPT_DIR" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
