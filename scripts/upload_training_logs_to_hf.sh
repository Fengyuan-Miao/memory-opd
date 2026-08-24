#!/usr/bin/env bash
# Upload OPD/MMem training logs to a Hugging Face dataset repository.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LOGS_DIR=${LOGS_DIR:-$REPO_ROOT/logs}
HF_REPO_ID=${HF_REPO_ID:-memory-r1/opd-mm-training-logs}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
HF_REVISION=${HF_REVISION:-main}
HF_PRIVATE=${HF_PRIVATE:-1}
HF_NUM_WORKERS=${HF_NUM_WORKERS:-8}
LATEST_DIR_ONLY=${LATEST_DIR_ONLY:-1}
LOG_SUBDIR=${LOG_SUBDIR:-}

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -d "$LOGS_DIR" ]]; then
  echo "Logs directory does not exist: $LOGS_DIR" >&2
  exit 1
fi

token=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
token_args=()
if [[ -n "$token" ]]; then
  token_args=(--token "$token")
elif ! hf auth whoami >/dev/null 2>&1; then
  echo "No Hugging Face authentication found." >&2
  echo "Set HF_TOKEN or run: hf auth login" >&2
  exit 1
fi

privacy_arg=--private
if [[ "$HF_PRIVATE" == "0" || "$HF_PRIVATE" == "false" ]]; then
  privacy_arg=--no-private
fi

include_args=()
selected_source=$LOGS_DIR
if [[ -n "$LOG_SUBDIR" ]]; then
  selected_dir=$LOGS_DIR/$LOG_SUBDIR
  if [[ ! -d "$selected_dir" ]]; then
    echo "Requested log subdirectory does not exist: $selected_dir" >&2
    exit 1
  fi
  include_args=(--include "$LOG_SUBDIR/*" --include "$LOG_SUBDIR/**")
  selected_source=$selected_dir
elif [[ "$LATEST_DIR_ONLY" != "0" && "$LATEST_DIR_ONLY" != "false" ]]; then
  latest_dir=$(find "$LOGS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr | sed -n '1p' | cut -d' ' -f2-)
  if [[ -z "$latest_dir" ]]; then
    echo "No subdirectories found under: $LOGS_DIR" >&2
    exit 1
  fi
  latest_name=${latest_dir#"$LOGS_DIR"/}
  include_args=(--include "$latest_name/*" --include "$latest_name/**")
  selected_source=$latest_dir
fi

echo "Uploading logs"
echo "  source:   $selected_source"
echo "  target:   https://huggingface.co/datasets/$HF_REPO_ID"
echo "  revision: $HF_REVISION"

# upload-large-folder hashes files locally, resumes interrupted uploads, and
# commits large directories in manageable batches. PID/lock/temp files are
# runtime state rather than experiment artifacts and are excluded.
hf upload-large-folder \
  "$HF_REPO_ID" \
  "$LOGS_DIR" \
  --repo-type "$HF_REPO_TYPE" \
  --revision "$HF_REVISION" \
  "$privacy_arg" \
  --num-workers "$HF_NUM_WORKERS" \
  "${include_args[@]}" \
  --exclude "*.pid" \
  --exclude "*.lock" \
  --exclude "*.tmp" \
  "${token_args[@]}"

echo "Upload completed: https://huggingface.co/datasets/$HF_REPO_ID"
