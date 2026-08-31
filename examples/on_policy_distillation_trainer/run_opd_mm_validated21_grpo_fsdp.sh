#!/usr/bin/env bash
# Validated21 MMem | GRPO with all-fail privileged state-level OPSD.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mmem/data/batches/validated21_final}
STORE_DIR=${STORE_DIR:-$DATASET_ROOT/opd_mm_store}
SPLIT_DIR=${SPLIT_DIR:-$STORE_DIR/subsets/grpo_holdout100_20260813}
MEMGALLERY_SPLIT_DIR=${MEMGALLERY_SPLIT_DIR:-$REPO_ROOT/dataset/mem_gallery/opd_mm_store/subsets/balanced_grpo_cap4_holdout100}
INDEX_GPU=${INDEX_GPU:-4}

build_index() {
    local name=$1
    if [[ -s "$STORE_DIR/indexes/$name/embeddings.npy" && -s "$STORE_DIR/indexes/$name/items.jsonl" ]]; then
        return
    fi
    CUDA_VISIBLE_DEVICES="$INDEX_GPU" python3 examples/data_preprocess/build_opd_mm_indexes_from_records.py \
        --records "$STORE_DIR/records.jsonl" \
        --output-dir "$STORE_DIR" \
        --index "$name" \
        --device cuda:0
}

build_index dense
build_index vision
build_index hybrid

if [[ ! -s "$SPLIT_DIR/train.parquet" || ! -s "$SPLIT_DIR/heldout_qas.jsonl" ]]; then
    python3 examples/data_preprocess/build_mem_gallery_opd_mm_train_subset.py \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$SPLIT_DIR" \
        --per-cell-cap 100000 \
        --seed 20260813 \
        --reserve-eval-samples 100 \
        --reserve-eval-seed 20260813 \
        --data-source opd_mm
fi

MMEM_VAL_PARQUET="$SPLIT_DIR/heldout_mmem_val.parquet"
MEMGALLERY_VAL_PARQUET="$SPLIT_DIR/heldout_memgallery_val.parquet"
if [[ ! -s "$MMEM_VAL_PARQUET" ]]; then
    python3 examples/data_preprocess/prepare_opd_mm_heldout_rlhf.py \
        --qas-jsonl "$SPLIT_DIR/heldout_qas.jsonl" \
        --output "$MMEM_VAL_PARQUET" \
        --dataset-root "$DATASET_ROOT" \
        --data-source opd_mm_mmem_val
fi
if [[ ! -s "$MEMGALLERY_VAL_PARQUET" ]]; then
    python3 examples/data_preprocess/prepare_opd_mm_heldout_rlhf.py \
        --qas-jsonl "$MEMGALLERY_SPLIT_DIR/heldout_qas.jsonl" \
        --output "$MEMGALLERY_VAL_PARQUET" \
        --dataset-root "$REPO_ROOT/dataset/mem_gallery" \
        --data-source opd_mm_memgallery_val
fi

export GRPO_DATA_DIR="$SPLIT_DIR"
export OPD_MM_TRAIN_FILES="['$SPLIT_DIR/train.parquet']"
export OPD_MM_VAL_FILES="['$MMEM_VAL_PARQUET','$MEMGALLERY_VAL_PARQUET']"
export OPD_MM_VECTOR_STORE_DIR="$STORE_DIR"

# Two visible GPUs host the student actor/rollout and two host the local 4B
# teacher. The answer/judge/INSPECT_EVIDENCE_IMAGE service is external, so all currently
# idle GPUs are used without touching unrelated processes on GPUs 0-3.
export TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
export TEACHER_GPU_MEMORY_UTIL=${TEACHER_GPU_MEMORY_UTIL:-0.80}
export OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-external}
export START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-0}
export OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://127.0.0.1:30803}
export OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-Qwen3.5-9B}
export OPD_MM_VERIFIER_BASE_URL=${OPD_MM_VERIFIER_BASE_URL:-$OUTCOME_SERVER_BASE_URL}
export OPD_MM_VERIFIER_MODEL=${OPD_MM_VERIFIER_MODEL:-$OUTCOME_SERVED_MODEL}
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}192.168.1.113"
export no_proxy="$NO_PROXY"
export ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}

export OPD_MM_KL_CREDIT_ENABLED=False
export OPD_MM_GRPO_ACTION_SELECTION=all_states
export OPD_MM_STATE_OPSD_ENABLED=True
export OPD_MM_STATE_OPSD_REWARD_GATED=True
export OPD_MM_REWARD_GATED_SFT_ENABLED=False
export DISTILLATION_ENABLED=True
export OPD_MM_ONLINE_SUPERVISION_MODE=opsd
export OPD_MM_RECORD_POLICY_STATES=1
export USE_REFERENCE_KL=${USE_REFERENCE_KL:-False}
export REFERENCE_KL_COEF=${REFERENCE_KL_COEF:-0.005}
export EVIDENCE_ANSWERABLE_WEIGHT=0.0
export EFFICIENCY_ACTION_FREE=3
export EFFICIENCY_ACTION_PENALTY=0.01
export EFFICIENCY_EVIDENCE_FREE=16
export EFFICIENCY_EVIDENCE_PENALTY=0.005
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-1000000}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
export RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL:-0}
export PROJECT_NAME=${PROJECT_NAME:-verl_grpo_opd_mm_validated21}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_validated21_grpo_allfail_opsd_answeronly_$(date +%Y%m%d_%H%M%S)}
# /home currently has less free space than one complete 4B optimizer
# checkpoint. Keep Ray spill files and the single final checkpoint on the
# root volume while preserving deterministic experiment names in W&B/logs.
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/memory-opd-ray}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/memory-opd-checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}

exec "$SCRIPT_DIR/run_opd_mm_grpo_fsdp.sh" "$@"
