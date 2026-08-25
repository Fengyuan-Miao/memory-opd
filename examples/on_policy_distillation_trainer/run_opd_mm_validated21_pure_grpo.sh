#!/usr/bin/env bash
# Validated21 MMem | pure GRPO, without OPD, OPSD, KL credit, or a teacher.
# GPU layout: 0-5 student/query encoders, 6-7 local Qwen3.5-9B service.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mmem/data/batches/validated21_final}
STORE_DIR=${STORE_DIR:-$DATASET_ROOT/opd_mm_store}
SPLIT_DIR=${SPLIT_DIR:-$STORE_DIR/subsets/grpo_holdout100_20260813}
MMEM_VAL_PARQUET=${MMEM_VAL_PARQUET:-$SPLIT_DIR/heldout_mmem_val.parquet}
MEMGALLERY_VAL_PARQUET=${MEMGALLERY_VAL_PARQUET:-$SPLIT_DIR/heldout_memgallery_val.parquet}

for required_path in \
    "$SPLIT_DIR/train.parquet" \
    "$MMEM_VAL_PARQUET" \
    "$MEMGALLERY_VAL_PARQUET" \
    "$STORE_DIR/indexes/dense/embeddings.npy" \
    "$STORE_DIR/indexes/vision/embeddings.npy" \
    "$STORE_DIR/indexes/hybrid/embeddings.npy"; do
    if [[ ! -s "$required_path" ]]; then
        echo "Missing required OPD-MM artifact: $required_path" >&2
        exit 1
    fi
done

export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
export MODEL_4B_PATH=${MODEL_4B_PATH:-$MODEL_ROOT/Qwen3.5-4B}
export MODEL_9B_PATH=${MODEL_9B_PATH:-$MODEL_ROOT/Qwen3.5-9B}
export OPD_MM_DENSE_MODEL_PATH=${OPD_MM_DENSE_MODEL_PATH:-$MODEL_ROOT/all-MiniLM-L6-v2}
export OPD_MM_VISION_MODEL_PATH=${OPD_MM_VISION_MODEL_PATH:-$MODEL_ROOT/SigLIP-Base-Patch16-384}
export OPD_MM_HYBRID_MODEL_PATH=${OPD_MM_HYBRID_MODEL_PATH:-$MODEL_ROOT/gme-Qwen2-VL-2B-Instruct}

for model_dir in \
    "$MODEL_4B_PATH" \
    "$MODEL_9B_PATH" \
    "$OPD_MM_DENSE_MODEL_PATH" \
    "$OPD_MM_VISION_MODEL_PATH" \
    "$OPD_MM_HYBRID_MODEL_PATH"; do
    if [[ ! -f "$model_dir/config.json" ]]; then
        echo "Missing model directory: $model_dir" >&2
        echo "Run: bash $SCRIPT_DIR/download_opd_mm_models.sh" >&2
        exit 1
    fi
done

export OPD_MODEL_PATH=${OPD_MODEL_PATH:-$MODEL_4B_PATH}
export GRPO_DATA_DIR="$SPLIT_DIR"
export OPD_MM_TRAIN_FILES="['$SPLIT_DIR/train.parquet']"
export OPD_MM_VAL_FILES="['$MMEM_VAL_PARQUET','$MEMGALLERY_VAL_PARQUET']"
export OPD_MM_VECTOR_STORE_DIR="$STORE_DIR"
export OPD_MM_REWARD_PATH=${OPD_MM_REWARD_PATH:-$REPO_ROOT/verl/experimental/opd_mm/outcome_reward.py}

# Six GPUs give the actor three TP=2 rollout replicas. Query encoders are
# distributed one worker per visible GPU instead of competing with the 9B VLM.
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export VERL_AGENT_LOOP_WORKER_CUDA_DEVICES=${VERL_AGENT_LOOP_WORKER_CUDA_DEVICES:-0,1,2,3,4,5}
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-6}
export ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}

# Pure GRPO: do not construct policy-state corrections or allocate a teacher.
export OPD_MM_KL_CREDIT_ENABLED=False
export OPD_MM_STATE_OPSD_ENABLED=False
export OPD_MM_STATE_OPSD_REWARD_GATED=False
export OPD_MM_REWARD_GATED_SFT_ENABLED=False
export OPD_MM_RECORD_POLICY_STATES=0
export OPD_MM_KL_CREDIT_ASSIGNMENT=0
export DISTILLATION_ENABLED=False
export USE_REFERENCE_KL=${USE_REFERENCE_KL:-False}

# n and batch sizes are global, not per GPU. This gives 48 trajectories per
# step while retaining lossless 16K prompt + 2K response budgets.
export ROLLOUT_N=${ROLLOUT_N:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}

# Keep the 9B context budget intact, but cap concurrency/KV-cache pressure.
export START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-1}
export OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-6,7}
export OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH:-$MODEL_9B_PATH}
export OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-Qwen3.5-9B}
export OUTCOME_SERVER_HOST=${OUTCOME_SERVER_HOST:-127.0.0.1}
export OUTCOME_SERVER_PORT=${OUTCOME_SERVER_PORT:-30803}
export OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://127.0.0.1:30803}
export OUTCOME_SERVER_TP=${OUTCOME_SERVER_TP:-2}
export OUTCOME_SERVER_GPU_MEMORY_UTIL=${OUTCOME_SERVER_GPU_MEMORY_UTIL:-0.60}
export OUTCOME_SERVER_MAX_MODEL_LEN=${OUTCOME_SERVER_MAX_MODEL_LEN:-40000}
export OUTCOME_SERVER_MAX_NUM_SEQS=${OUTCOME_SERVER_MAX_NUM_SEQS:-8}
export OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS=${OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}
export OPD_MM_VERIFIER_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_VERIFIER_MODEL="$OUTCOME_SERVED_MODEL"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

# Match the previous reward and dual-validation setup; only OPD is removed.
export EVIDENCE_ANSWERABLE_WEIGHT=${EVIDENCE_ANSWERABLE_WEIGHT:-0.2}
export EFFICIENCY_ACTION_FREE=${EFFICIENCY_ACTION_FREE:-3}
export EFFICIENCY_ACTION_PENALTY=${EFFICIENCY_ACTION_PENALTY:-0.01}
export EFFICIENCY_EVIDENCE_FREE=${EFFICIENCY_EVIDENCE_FREE:-16}
export EFFICIENCY_EVIDENCE_PENALTY=${EFFICIENCY_EVIDENCE_PENALTY:-0.005}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-10}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-1000000}
export RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL:-0}
export PROJECT_NAME=${PROJECT_NAME:-verl_grpo_opd_mm_validated21}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_validated21_pure_grpo_$(date +%Y%m%d_%H%M%S)}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/verl_ray_$(id -u)}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/memory-opd-checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-mmem}
export WANDB_VAL_CASES=${WANDB_VAL_CASES:-0}
export WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS:-True}
export WANDB_GPU_MEMORY_METRICS=${WANDB_GPU_MEMORY_METRICS:-True}

exec bash "$SCRIPT_DIR/run_opd_mm_grpo_fsdp.sh" "$@"
