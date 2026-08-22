#!/usr/bin/env bash
# Validated21 MMem | pure online privileged distillation with GPU query encoders.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mmem/batches/validated21_final}
STORE_DIR=${STORE_DIR:-$DATASET_ROOT/opd_mm_store}
SPLIT_DIR=${SPLIT_DIR:-$STORE_DIR/subsets/grpo_holdout100_20260813}

MMEM_VAL_PARQUET=${MMEM_VAL_PARQUET:-$SPLIT_DIR/heldout_mmem_val.parquet}
MEMGALLERY_VAL_PARQUET=${MEMGALLERY_VAL_PARQUET:-$SPLIT_DIR/heldout_memgallery_val.parquet}
for required_path in \
    "$SPLIT_DIR/train.parquet" \
    "$SPLIT_DIR/heldout_qas.jsonl" \
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

export DATA_DIR="$SPLIT_DIR"
export OPD_MM_DATASET_ROOT="$DATASET_ROOT"
export OPD_MM_TRAIN_FILES="['$SPLIT_DIR/train.parquet']"
export OPD_MM_HELDOUT_QAS="$SPLIT_DIR/heldout_qas.jsonl"
export OPD_MM_VAL_PARQUET="$MMEM_VAL_PARQUET"
export OPD_MM_VAL_FILES="['$MMEM_VAL_PARQUET','$MEMGALLERY_VAL_PARQUET']"
export OPD_MM_VECTOR_STORE_DIR="$STORE_DIR"

# GPUs 0-5 host the student and colocated query encoders. GPUs 6-7 host the
# frozen 4B teacher. Verifier, answer, judge, and INSPECT_RAW all use the
# external 9B vLLM; INSPECT_RAW sends complete images as base64 data URLs.
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
export START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-0}
export OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-external}
export VERL_AGENT_LOOP_WORKER_CUDA_DEVICES=${VERL_AGENT_LOOP_WORKER_CUDA_DEVICES:-0,1,2,3,4,5}
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-6}
export ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}

export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
export STUDENT_MODEL=${STUDENT_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}
export TEACHER_GPU_MEMORY_UTIL=${TEACHER_GPU_MEMORY_UTIL:-0.80}
export TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}
export OPD_MM_VERIFIER_BASE_URL=${OPD_MM_VERIFIER_BASE_URL:-http://127.0.0.1:30803}
export OPD_MM_VERIFIER_MODEL=${OPD_MM_VERIFIER_MODEL:-Qwen3.5-9B}
export OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://127.0.0.1:30803}
export OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-Qwen3.5-9B}
export OPD_MM_RAW_INSPECTOR_TIMEOUT=${OPD_MM_RAW_INSPECTOR_TIMEOUT:-120}
export OPD_MM_RAW_INSPECTOR_BYPASS_PROXY=${OPD_MM_RAW_INSPECTOR_BYPASS_PROXY:-1}
export OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE=${OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE:-$DATASET_ROOT/image/Museum_Urban_Nature_Cat_London_Exhibition_Community_Life/D8_IMG_003.jpg}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-50}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-1000000}
export PROJECT_NAME=${PROJECT_NAME:-verl_opsd_opd_mm_validated21}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_validated21_pure_opsd_gpuencoder_$(date +%Y%m%d_%H%M%S)}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-$REPO_ROOT/.runtime/ray}

exec bash "$SCRIPT_DIR/run_opd_mm_opsd_fsdp.sh" "$@"
