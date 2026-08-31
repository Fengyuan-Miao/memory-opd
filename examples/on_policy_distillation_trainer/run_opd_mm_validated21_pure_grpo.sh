#!/usr/bin/env bash
# Teacher-success Mem-Gallery subset | pure GRPO, without OPD, OPSD, KL credit, or a teacher.
# Reference-policy KL is retained only as a policy-collapse regularizer.
# Portable 6-GPU layout:
#   GPUs 0-3: 4B student, reference policy, and query encoders
#   GPUs 4-5: local 9B verifier / selector / answer / judge / image inspector

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mem_gallery}
STORE_DIR=${STORE_DIR:-$DATASET_ROOT/opd_mm_store}
SUBSET_NAME=${SUBSET_NAME:-teacher_success_full_minus_heldout100_20260828}
SPLIT_DIR=${SPLIT_DIR:-$STORE_DIR/subsets/$SUBSET_NAME}
MEMGALLERY_HELDOUT_DIR=${MEMGALLERY_HELDOUT_DIR:-$STORE_DIR/subsets/balanced_grpo_cap4_holdout100}
if [[ -z "${MMEM_DATASET_ROOT:-}" ]]; then
    for candidate in \
        "$REPO_ROOT/dataset/mmem/batches/validated21_final" \
        "$REPO_ROOT/dataset/mmem/data/batches/validated21_final"; do
        if [[ -d "$candidate/opd_mm_store" ]]; then
            MMEM_DATASET_ROOT=$candidate
            break
        fi
    done
    MMEM_DATASET_ROOT=${MMEM_DATASET_ROOT:-$REPO_ROOT/dataset/mmem/batches/validated21_final}
fi
MMEM_STORE_DIR=${MMEM_STORE_DIR:-$MMEM_DATASET_ROOT/opd_mm_store}
MMEM_HELDOUT_DIR=${MMEM_HELDOUT_DIR:-$MMEM_STORE_DIR/subsets/grpo_holdout100_20260813}
MEMGALLERY_VAL_PARQUET=${MEMGALLERY_VAL_PARQUET:-$MMEM_HELDOUT_DIR/heldout_memgallery_val.parquet}
MMEM_VAL_PARQUET=${MMEM_VAL_PARQUET:-$MMEM_HELDOUT_DIR/heldout_mmem_val.parquet}

export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
export MODEL_4B_PATH=${MODEL_4B_PATH:-$MODEL_ROOT/Qwen3.5-4B}
export MODEL_9B_PATH=${MODEL_9B_PATH:-$MODEL_ROOT/Qwen3.5-9B}
export OPD_MM_DENSE_MODEL_PATH=${OPD_MM_DENSE_MODEL_PATH:-$MODEL_ROOT/all-MiniLM-L6-v2}
export OPD_MM_VISION_MODEL_PATH=${OPD_MM_VISION_MODEL_PATH:-$MODEL_ROOT/SigLIP-Base-Patch16-384}
export OPD_MM_HYBRID_MODEL_PATH=${OPD_MM_HYBRID_MODEL_PATH:-$MODEL_ROOT/gme-Qwen2-VL-2B-Instruct}

# Match the portable OPD launcher: download public models when they are not
# already cached under MODEL_ROOT.
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

# Fetch only the previously filtered training subset when the base dataset is
# already present on a new machine.
if [[ ! -s "$SPLIT_DIR/train.parquet" ]]; then
    DATASET_ROOT="$DATASET_ROOT" SUBSET_NAME="$SUBSET_NAME" \
        bash "$REPO_ROOT/scripts/download_mem_gallery_subset.sh"
fi

for required_path in \
    "$SPLIT_DIR/train.parquet" \
    "$MEMGALLERY_HELDOUT_DIR/heldout_qas.jsonl" \
    "$MEMGALLERY_VAL_PARQUET" \
    "$MMEM_VAL_PARQUET" \
    "$STORE_DIR/records.jsonl" \
    "$STORE_DIR/indexes/dense/embeddings.npy" \
    "$STORE_DIR/indexes/vision/embeddings.npy" \
    "$STORE_DIR/indexes/hybrid/embeddings.npy" \
    "$MMEM_STORE_DIR/records.jsonl" \
    "$MMEM_STORE_DIR/indexes/dense/embeddings.npy" \
    "$MMEM_STORE_DIR/indexes/vision/embeddings.npy" \
    "$MMEM_STORE_DIR/indexes/hybrid/embeddings.npy"; do
    if [[ ! -s "$required_path" ]]; then
        echo "Missing required OPD-MM artifact: $required_path" >&2
        exit 1
    fi
done

export OPD_MODEL_PATH=${OPD_MODEL_PATH:-$MODEL_4B_PATH}
export GRPO_DATA_DIR="$SPLIT_DIR"
export OPD_MM_DATASET_ROOT="$DATASET_ROOT"
export OPD_MM_DATASET_ROOTS=${OPD_MM_DATASET_ROOTS:-$DATASET_ROOT:$MMEM_DATASET_ROOT}
export OPD_MM_TRAIN_FILES="['$SPLIT_DIR/train.parquet']"
export OPD_MM_VAL_FILES="['$MEMGALLERY_VAL_PARQUET','$MMEM_VAL_PARQUET']"
export OPD_MM_HELDOUT_QAS="$MEMGALLERY_HELDOUT_DIR/heldout_qas.jsonl"
export OPD_MM_VAL_PARQUET="$MEMGALLERY_VAL_PARQUET"
export OPD_MM_VECTOR_STORE_DIR="$STORE_DIR"
export OPD_MM_REWARD_PATH=${OPD_MM_REWARD_PATH:-$REPO_ROOT/verl/experimental/opd_mm/outcome_reward.py}

# Ray sees only GPUs 0-3. The local 9B service is launched separately on 4-5.
export TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export VERL_AGENT_LOOP_WORKER_CUDA_DEVICES=${VERL_AGENT_LOOP_WORKER_CUDA_DEVICES:-0,1,2,3}
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-4}
# Qwen3.5-4B has 16 attention heads, so SP=4 is valid on the four student GPUs.
export ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-4}
export ROLLOUT_TP=${ROLLOUT_TP:-2}

# Pure GRPO: do not construct policy-state corrections or allocate a teacher.
export OPD_MM_KL_CREDIT_ENABLED=False
export OPD_MM_STATE_OPSD_ENABLED=False
export OPD_MM_STATE_OPSD_REWARD_GATED=False
export OPD_MM_REWARD_GATED_SFT_ENABLED=False
export OPD_MM_RECORD_POLICY_STATES=0
export OPD_MM_KL_CREDIT_ASSIGNMENT=0
export DISTILLATION_ENABLED=False
export USE_REFERENCE_KL=${USE_REFERENCE_KL:-True}
export REFERENCE_KL_COEF=${REFERENCE_KL_COEF:-0.005}
export ACTOR_LOSS_AGG_MODE=${ACTOR_LOSS_AGG_MODE:-seq-mean-token-mean}
export ACTOR_LR=${ACTOR_LR:-2e-7}

# n and batch sizes are global, not per GPU. Eight prompts with n=4 give 32
# trajectories, matching the old per-GPU rollout load on four 80-GiB GPUs.
export ROLLOUT_N=${ROLLOUT_N:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}

# Keep the 9B context budget intact, but cap concurrency/KV-cache pressure.
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
export OPD_MM_VERIFIER_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_VERIFIER_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_RAW_INSPECTOR_TIMEOUT=${OPD_MM_RAW_INSPECTOR_TIMEOUT:-120}
export OPD_MM_RAW_INSPECTOR_BYPASS_PROXY=${OPD_MM_RAW_INSPECTOR_BYPASS_PROXY:-1}
export OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE=${OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE:-$DATASET_ROOT/data/image/Academic_Animal_Pet_Research_Life/D8_IMG_003.jpg}
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

# Match the previous reward and dual-validation setup; only OPD is removed.
export EVIDENCE_ANSWERABLE_WEIGHT=${EVIDENCE_ANSWERABLE_WEIGHT:-0.0}
export REPEAT_PENALTY=${REPEAT_PENALTY:-0.05}
export MAX_ACTION_PENALTY=${MAX_ACTION_PENALTY:-0.2}
export ERROR_PENALTY=${ERROR_PENALTY:-0.5}
export NON_STOP_PENALTY=${NON_STOP_PENALTY:-0.3}
export EFFICIENCY_ACTION_FREE=${EFFICIENCY_ACTION_FREE:-3}
export EFFICIENCY_ACTION_PENALTY=${EFFICIENCY_ACTION_PENALTY:-0.01}
export EFFICIENCY_EVIDENCE_FREE=${EFFICIENCY_EVIDENCE_FREE:-16}
export EFFICIENCY_EVIDENCE_PENALTY=${EFFICIENCY_EVIDENCE_PENALTY:-0.005}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
export TEST_FREQ=${TEST_FREQ:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export SAVE_FREQ=${SAVE_FREQ:-1000000}
export RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL:-0}
export PROJECT_NAME=${PROJECT_NAME:-verl_grpo_opd_mm_memgallery}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_memgallery_teacher_success1239_pure_grpo_$(date +%Y%m%d_%H%M%S)}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/verl_ray_$(id -u)}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/memory-opd-checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-mmem}
export WANDB_VAL_CASES=${WANDB_VAL_CASES:-0}
# Enable W&B's native GPU telemetry on the remote machine.
export WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS:-False}
# Capture the trainer/Ray console stream in the W&B Logs tab.
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

exec bash "$SCRIPT_DIR/run_opd_mm_grpo_fsdp.sh" "$@"
