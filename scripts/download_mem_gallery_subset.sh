#!/usr/bin/env bash
# Download the teacher-success Mem-Gallery training subset into this checkout.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

HF_REPO_ID=${HF_REPO_ID:-memory-rl/Mem-Gallery}
HF_REVISION=${HF_REVISION:-main}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-8}
SUBSET_NAME=${SUBSET_NAME:-teacher_success_full_minus_heldout100_20260828}
REMOTE_SUBDIR=${REMOTE_SUBDIR:-opd_mm_store/subsets/$SUBSET_NAME}
DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mem_gallery}
TARGET_DIR=$DATASET_ROOT/$REMOTE_SUBDIR

for command_name in hf python3; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing '$command_name'. Install with: pip install -U huggingface_hub" >&2
        exit 1
    fi
done

token=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
token_args=()
if [[ -n "$token" ]]; then
    token_args=(--token "$token")
elif ! hf auth whoami >/dev/null 2>&1; then
    echo "Hugging Face authentication is required for private repo $HF_REPO_ID." >&2
    echo "Set HF_TOKEN or run: hf auth login" >&2
    exit 1
fi

mkdir -p "$DATASET_ROOT"

echo "Downloading hf://datasets/$HF_REPO_ID/$REMOTE_SUBDIR"
echo "Target: $TARGET_DIR"
hf download "$HF_REPO_ID" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --include "$REMOTE_SUBDIR/*" \
    --include "$REMOTE_SUBDIR/**" \
    --local-dir "$DATASET_ROOT" \
    --max-workers "$HF_MAX_WORKERS" \
    "${token_args[@]}"

for required_file in \
    manifest.json \
    train_qas.jsonl \
    train_sample_ids.txt \
    rejected_sample_ids.txt \
    train.jsonl \
    train.parquet; do
    if [[ ! -s "$TARGET_DIR/$required_file" ]]; then
        echo "Missing downloaded subset file: $TARGET_DIR/$required_file" >&2
        exit 1
    fi
done

TARGET_PARQUET="$TARGET_DIR/train.parquet" python3 - <<'PY'
import os

import pyarrow.parquet as pq

path = os.environ["TARGET_PARQUET"]
row_count = pq.read_metadata(path).num_rows
if row_count != 1239:
    raise SystemExit(f"Unexpected training row count in {path}: {row_count} (expected 1239)")
print(f"Mem-Gallery teacher-success subset is ready: {path} ({row_count} rows)")
PY
