#!/usr/bin/env bash

set -euo pipefail

HF_REPO=${HF_REPO:-memory-rl/MMEM}
HF_REVISION=${HF_REVISION:-main}
DATASET_DIR=${DATASET_DIR:-/dataset}
ARCHIVE_NAME=${ARCHIVE_NAME:-MMEM.tar.gz}
EXPECTED_SHA256=${EXPECTED_SHA256:-1ffabf39d3536b8bed6ceed941ec91dcb3640c61d14d0ca262dce303edf7bdd3}

ARCHIVE_PATH="$DATASET_DIR/$ARCHIVE_NAME"
EXTRACTED_ROOT="$DATASET_DIR/mmem/batches/validated21_final"

for command_name in hf tar sha256sum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
done

mkdir -p "$DATASET_DIR"

echo "Downloading hf://datasets/$HF_REPO/$ARCHIVE_NAME to $ARCHIVE_PATH"
hf download "$HF_REPO" "$ARCHIVE_NAME" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --local-dir "$DATASET_DIR"

actual_sha256=$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Checksum mismatch for $ARCHIVE_PATH" >&2
    echo "expected: $EXPECTED_SHA256" >&2
    echo "actual:   $actual_sha256" >&2
    exit 1
fi

echo "Extracting into $DATASET_DIR (existing files will be overwritten)"
tar -xzf "$ARCHIVE_PATH" \
    -C "$DATASET_DIR" \
    --overwrite \
    --no-same-owner

for required_dir in dialog image opd_mm_store reports; do
    if [[ ! -d "$EXTRACTED_ROOT/$required_dir" ]]; then
        echo "Missing extracted directory: $EXTRACTED_ROOT/$required_dir" >&2
        exit 1
    fi
done

required_train_parquet="$EXTRACTED_ROOT/opd_mm_store/subsets/grpo_holdout100_20260813/train.parquet"
if [[ ! -f "$required_train_parquet" ]]; then
    echo "Missing required training split: $required_train_parquet" >&2
    exit 1
fi

echo "MMEM is ready at $EXTRACTED_ROOT"
