#!/bin/bash
# DreamZero find9shape_small LoRA fine-tuning with annotation-defined context starts.
#
# Usage:
#   bash scripts/train/find9shape_annotation_context_training.sh --debug 2>&1 | tee logs/find9shape_train_context_$(date +%Y%m%d_%H%M%S).log
#
# Common overrides:
#   NUM_GPUS=1 MAX_STEPS=1000 OUTPUT_DIR=/path/to/output \
#   FIND9SHAPE_ANNOTATION_CSV=/path/to/annotation.csv \
#     bash scripts/train/find9shape_annotation_context_training.sh

set -e

export HYDRA_FULL_ERROR=1

DEBUG_ACTION_ERROR_METRICS=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_ACTION_ERROR_METRICS=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

REMOTE_PROJECT_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero"
REMOTE_DATASET_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/dreamzero/datasets"
REMOTE_CHECKPOINT_ROOT="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero/checkpoints"
VENV_DIR=${VENV_DIR:-"$REMOTE_PROJECT_ROOT/.venv"}

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

FIND9SHAPE_DATA_ROOT=${FIND9SHAPE_DATA_ROOT:-"/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/dreamzero/datasets/find_imposter_shape_9_vla_v0_raw"}
FIND9SHAPE_ANNOTATION_CSV=${FIND9SHAPE_ANNOTATION_CSV:-"$PWD/annotation.csv"}
FIND9SHAPE_GOAL_IMAGE_ROOT=${FIND9SHAPE_GOAL_IMAGE_ROOT:-"$FIND9SHAPE_DATA_ROOT/goal_images"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REMOTE_CHECKPOINT_ROOT/dreamzero_find9shape_annotation_context_lora_raw"}

if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$REMOTE_CHECKPOINT_ROOT/Wan2.1-I2V-14B-480P"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$REMOTE_CHECKPOINT_ROOT/umt5-xxl"}
# AGIBOT_CKPT_DIR=${AGIBOT_CKPT_DIR:-"$REMOTE_CHECKPOINT_ROOT/DreamZero-AgiBot"}
AGIBOT_CKPT_DIR=${AGIBOT_CKPT_DIR:-"$REMOTE_CHECKPOINT_ROOT/dreamzero_find9shape_annotation_context_lora_raw/checkpoint-10000"} 
HF_CACHE_ROOT=${SLURM_TMPDIR:-/tmp}
export HF_HOME=${HF_HOME:-"$HF_CACHE_ROOT/hf_cache_${USER:-zhiyanli}"}

MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}

if [ ! -d "$FIND9SHAPE_DATA_ROOT" ]; then
    echo "ERROR: find9shape dataset not found at $FIND9SHAPE_DATA_ROOT"
    echo "Set FIND9SHAPE_DATA_ROOT to your LeRobot-format find9shape dataset."
    exit 1
fi

if [ ! -f "$FIND9SHAPE_ANNOTATION_CSV" ]; then
    echo "ERROR: annotation CSV not found at $FIND9SHAPE_ANNOTATION_CSV"
    echo "Set FIND9SHAPE_ANNOTATION_CSV to annotation.csv."
    exit 1
fi

if [ ! -d "$FIND9SHAPE_GOAL_IMAGE_ROOT" ]; then
    echo "ERROR: goal image root not found at $FIND9SHAPE_GOAL_IMAGE_ROOT"
    echo "Generate it with:"
    echo "  python scripts/data/extract_lerobot_goal_images.py --dataset-root $FIND9SHAPE_DATA_ROOT --output-root $FIND9SHAPE_GOAL_IMAGE_ROOT"
    echo "Or set FIND9SHAPE_GOAL_IMAGE_ROOT to an existing generated-goal folder."
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
    report_to=tensorboard \
    data=dreamzero/find9shape_annotation_context \
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
    find9shape_annotation_csv="$FIND9SHAPE_ANNOTATION_CSV" \
    find9shape_goal_image_root="$FIND9SHAPE_GOAL_IMAGE_ROOT" \
    "${WAN_ARGS[@]}" \
    tokenizer_path="$TOKENIZER_PATH" \
    pretrained_model_path="$AGIBOT_CKPT_DIR" \
    action_head_cfg.config.debug_action_error_metrics="$DEBUG_ACTION_ERROR_METRICS" \
    action_head_cfg.config.use_goal_image_conditioning=true \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    "${EXTRA_ARGS[@]}"
