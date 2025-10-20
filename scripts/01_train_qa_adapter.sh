#!/bin/bash
# This script runs the SFT for the OpenBookQA task.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# We read parameters from our config.yaml, but for clarity in this
# first script, we'll define them here. In a more advanced setup,
# you'd use a tool to parse the YAML in bash.
export WANDB_PROJECT="llm-alignment-pipeline"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASK_NAME="openbookqa"
DATASET_NAME="openbookqa"
DATASET_SUBSET="main"
ADAPTER_DIR="models/adapters"
LR=2.0e-4
MAX_STEPS=500
BATCH_SIZE=4
GRAD_ACCUM=2

# --- Run Training ---
echo "Starting SFT for $TASK_NAME..."

python src/training/run_sft.py \
    --task_name $TASK_NAME \
    --base_model_name $BASE_MODEL \
    --dataset_name $DATASET_NAME \
    --dataset_subset $DATASET_SUBSET \
    --output_adapter_dir $ADAPTER_DIR \
    --learning_rate $LR \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --wandb_project $WANDB_PROJECT

echo "SFT for $TASK_NAME complete. Adapter saved in $ADAPTER_DIR/$TASK_NAME"