#!/usr/bin/env bash
# Mem-Gallery (1239 teacher-success train) | pure online privileged distillation.
#
# Portable 6-GPU layout:
#   GPUs 0-2: 4B student and query encoders
#   GPU 3: frozen 4B teacher
#   GPUs 4-5: local 9B verifier / selector / answer / judge / INSPECT_EVIDENCE_IMAGE

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
export MODEL_4B_PATH=${MODEL_4B_PATH:-$MODEL_ROOT/Qwen3.5-4B}
export MODEL_9B_PATH=${MODEL_9B_PATH:-$MODEL_ROOT/Qwen3.5-9B}
export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_4B_PATH}
export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_4B_PATH}
export OPD_MM_DENSE_MODEL_PATH=${OPD_MM_DENSE_MODEL_PATH:-$MODEL_ROOT/all-MiniLM-L6-v2}
export OPD_MM_VISION_MODEL_PATH=${OPD_MM_VISION_MODEL_PATH:-$MODEL_ROOT/SigLIP-Base-Patch16-384}
export OPD_MM_HYBRID_MODEL_PATH=${OPD_MM_HYBRID_MODEL_PATH:-$MODEL_ROOT/gme-Qwen2-VL-2B-Instruct}

# A fresh machine can download all five public model repositories before data
# preparation. Set AUTO_DOWNLOAD_MODELS=0 when model paths are mounted instead.
if [[ "${AUTO_DOWNLOAD_MODELS:-1}" == "1" ]]; then
    MODEL_ROOT="$MODEL_ROOT" \
    MODEL_4B_PATH="$MODEL_4B_PATH" \
    MODEL_9B_PATH="$MODEL_9B_PATH" \
    OPD_MM_DENSE_MODEL_PATH="$OPD_MM_DENSE_MODEL_PATH" \
    OPD_MM_VISION_MODEL_PATH="$OPD_MM_VISION_MODEL_PATH" \
    OPD_MM_HYBRID_MODEL_PATH="$OPD_MM_HYBRID_MODEL_PATH" \
        bash "$SCRIPT_DIR/download_opd_mm_models.sh"
fi

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

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mem_gallery}
STORE_DIR=${STORE_DIR:-$DATASET_ROOT/opd_mm_store}
SUBSET_NAME=${SUBSET_NAME:-teacher_success_full_minus_heldout100_20260828}
SPLIT_DIR=${SPLIT_DIR:-$STORE_DIR/subsets/$SUBSET_NAME}
MEMGALLERY_HELDOUT_DIR=${MEMGALLERY_HELDOUT_DIR:-$STORE_DIR/subsets/balanced_grpo_cap4_holdout100}
MMEM_DATASET_ROOT=${MMEM_DATASET_ROOT:-$REPO_ROOT/dataset/mmem/batches/validated21_final}
MMEM_STORE_DIR=${MMEM_STORE_DIR:-$MMEM_DATASET_ROOT/opd_mm_store}
MMEM_HELDOUT_DIR=${MMEM_HELDOUT_DIR:-$MMEM_STORE_DIR/subsets/grpo_holdout100_20260813}
MEMGALLERY_VAL_PARQUET=${MEMGALLERY_VAL_PARQUET:-$MMEM_HELDOUT_DIR/heldout_memgallery_val.parquet}
MMEM_VAL_PARQUET=${MMEM_VAL_PARQUET:-$MMEM_HELDOUT_DIR/heldout_mmem_val.parquet}

# Reuse the exact Mem-Gallery store and three indexes used by the existing
# evaluation pipeline. They are dataset artifacts, not per-experiment outputs.
for required_path in \
    "$DATASET_ROOT/data/dialog" \
    "$DATASET_ROOT/data/image" \
    "$STORE_DIR/records.jsonl" \
    "$STORE_DIR/qas.jsonl" \
    "$STORE_DIR/indexes/dense/embeddings.npy" \
    "$STORE_DIR/indexes/vision/embeddings.npy" \
    "$STORE_DIR/indexes/hybrid/embeddings.npy"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required Mem-Gallery dataset artifact: $required_path" >&2
        echo "Place the prepared Mem-Gallery dataset under: $DATASET_ROOT" >&2
        exit 1
    fi
done

# Download the teacher-success split when the base Mem-Gallery dataset is
# present but the filtered training artifact has not been fetched yet.
if [[ ! -s "$SPLIT_DIR/train.parquet" ]]; then
    DATASET_ROOT="$DATASET_ROOT" SUBSET_NAME="$SUBSET_NAME" \
        bash "$REPO_ROOT/scripts/download_mem_gallery_subset.sh"
fi

for required_path in \
    "$SPLIT_DIR/train.parquet" \
    "$MEMGALLERY_HELDOUT_DIR/heldout_qas.jsonl" \
    "$MEMGALLERY_VAL_PARQUET" \
    "$MMEM_VAL_PARQUET" \
    "$MMEM_STORE_DIR/records.jsonl" \
    "$MMEM_STORE_DIR/indexes/dense/embeddings.npy" \
    "$MMEM_STORE_DIR/indexes/vision/embeddings.npy" \
    "$MMEM_STORE_DIR/indexes/hybrid/embeddings.npy" \
    "$STORE_DIR/indexes/dense/embeddings.npy" \
    "$STORE_DIR/indexes/vision/embeddings.npy" \
    "$STORE_DIR/indexes/hybrid/embeddings.npy"; do
    if [[ ! -s "$required_path" ]]; then
        echo "Missing required Mem-Gallery artifact: $required_path" >&2
        exit 1
    fi
done

export DATA_DIR="$SPLIT_DIR"
export OPD_MM_DATASET_ROOT="$DATASET_ROOT"
export OPD_MM_VECTOR_STORE_DIR="$STORE_DIR"
export OPD_MM_TRAIN_FILES="['$SPLIT_DIR/train.parquet']"
export OPD_MM_HELDOUT_QAS="$MEMGALLERY_HELDOUT_DIR/heldout_qas.jsonl"
export OPD_MM_VAL_PARQUET="$MEMGALLERY_VAL_PARQUET"
export OPD_MM_VAL_FILES="['$MEMGALLERY_VAL_PARQUET','$MMEM_VAL_PARQUET']"

# Ray sees GPUs 0-3: three student workers and one frozen-teacher worker.
# The local 9B service is launched separately on physical GPUs 4-5.
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-1}
export TEACHER_TP=${TEACHER_TP:-1}
export VERL_AGENT_LOOP_WORKER_CUDA_DEVICES=${VERL_AGENT_LOOP_WORKER_CUDA_DEVICES:-0,1,2}
# Qwen3.5-4B has 16 attention heads. Ulysses SP must divide 16, so a
# three-worker student pool cannot use SP=3; retain FSDP data parallelism and
# keep sequence parallelism disabled.
export ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}

export START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-1}
export OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-4,5}
export OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH:-$MODEL_9B_PATH}
export OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-Qwen3.5-9B}
export OUTCOME_SERVER_HOST=${OUTCOME_SERVER_HOST:-127.0.0.1}
export OUTCOME_SERVER_PORT=${OUTCOME_SERVER_PORT:-30803}
export OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://127.0.0.1:30803}
export OUTCOME_SERVER_TP=${OUTCOME_SERVER_TP:-2}
export OUTCOME_SERVER_GPU_MEMORY_UTIL=${OUTCOME_SERVER_GPU_MEMORY_UTIL:-0.70}
export OUTCOME_SERVER_MAX_MODEL_LEN=${OUTCOME_SERVER_MAX_MODEL_LEN:-40000}
export OUTCOME_SERVER_MAX_NUM_SEQS=${OUTCOME_SERVER_MAX_NUM_SEQS:-16}
export OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS=${OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS:-16384}
export OPD_MM_VERIFIER_BASE_URL=${OPD_MM_VERIFIER_BASE_URL:-$OUTCOME_SERVER_BASE_URL}
export OPD_MM_VERIFIER_MODEL=${OPD_MM_VERIFIER_MODEL:-$OUTCOME_SERVED_MODEL}
export OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE=${OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE:-$DATASET_ROOT/data/image/Academic_Animal_Pet_Research_Life/D8_IMG_003.jpg}
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export VAL_ROLLOUT_DO_SAMPLE=${VAL_ROLLOUT_DO_SAMPLE:-False}
# DataProto currently requires equal worker chunks. Six workers divide the
# global train batch of 24 exactly and map evenly to the three student GPUs.
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-6}
export REWARD_WORKERS=${REWARD_WORKERS:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export TEACHER_GPU_MEMORY_UTIL=${TEACHER_GPU_MEMORY_UTIL:-0.65}
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}
export TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
export TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-16}
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-50}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-1000000}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-mmem}
export WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS:-True}
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}
export WANDB_VAL_CASES=${WANDB_VAL_CASES:-0}
export WANDB_VAL_ACTIONS_ONLY=${WANDB_VAL_ACTIONS_ONLY:-True}
export PROJECT_NAME=${PROJECT_NAME:-verl_opsd_opd_mm_memgallery}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_memgallery_teacher_success1239_pure_opsd_$(date +%Y%m%d_%H%M%S)}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/verl_ray_$(id -u)}

if (( TRAIN_BATCH_SIZE % AGENT_LOOP_NUM_WORKERS != 0 )); then
    echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by AGENT_LOOP_NUM_WORKERS=$AGENT_LOOP_NUM_WORKERS" >&2
    exit 1
fi

exec bash "$SCRIPT_DIR/run_opd_mm_opsd_fsdp.sh" "$@"
