#!/usr/bin/env bash
# Upload OPD/MMem training logs to a Hugging Face dataset repository.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LOGS_DIR=${LOGS_DIR:-$REPO_ROOT/logs}
HF_REPO_ID=${HF_REPO_ID:-memory-rl/opd-mm-training-logs}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
HF_REVISION=${HF_REVISION:-main}
HF_PRIVATE=${HF_PRIVATE:-1}
HF_NUM_WORKERS=${HF_NUM_WORKERS:-8}
LOG_SUBDIR=${LOG_SUBDIR:-}

for command_name in hf python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing '$command_name'. Install huggingface_hub with: pip install -U huggingface_hub" >&2
    exit 1
  fi
done

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

remote_dir_output=$(
  HF_SYNC_TOKEN="$token" \
  HF_SYNC_REPO_ID="$HF_REPO_ID" \
  HF_SYNC_REPO_TYPE="$HF_REPO_TYPE" \
  HF_SYNC_REVISION="$HF_REVISION" \
  python3 - <<'PY'
import os

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFolder

for entry in HfApi().list_repo_tree(
    repo_id=os.environ["HF_SYNC_REPO_ID"],
    repo_type=os.environ["HF_SYNC_REPO_TYPE"],
    revision=os.environ["HF_SYNC_REVISION"],
    token=os.environ.get("HF_SYNC_TOKEN") or None,
):
    if isinstance(entry, RepoFolder):
        print(entry.path)
PY
)
remote_dirs=()
if [[ -n "$remote_dir_output" ]]; then
  mapfile -t remote_dirs <<<"$remote_dir_output"
fi

declare -A remote_dir_set=()
for directory in "${remote_dirs[@]}"; do
  remote_dir_set["$directory"]=1
done

selected_dirs=()
if [[ -n "$LOG_SUBDIR" ]]; then
  if [[ "$LOG_SUBDIR" == /* || "$LOG_SUBDIR" == *".."* || "$LOG_SUBDIR" == */* ]]; then
    echo "LOG_SUBDIR must be a single relative directory name: $LOG_SUBDIR" >&2
    exit 1
  fi
  selected_dir=$LOGS_DIR/$LOG_SUBDIR
  if [[ ! -d "$selected_dir" ]]; then
    echo "Requested log subdirectory does not exist: $selected_dir" >&2
    exit 1
  fi
  if [[ -z "${remote_dir_set[$LOG_SUBDIR]+x}" ]]; then
    selected_dirs+=("$LOG_SUBDIR")
  fi
else
  mapfile -t local_dirs < <(
    find "$LOGS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name .cache -printf '%f\n' | sort
  )
  for directory in "${local_dirs[@]}"; do
    if [[ -z "${remote_dir_set[$directory]+x}" ]]; then
      selected_dirs+=("$directory")
    fi
  done
fi

if (( ${#selected_dirs[@]} == 0 )); then
  echo "No new local log directories to upload."
  exit 0
fi

include_args=()
for directory in "${selected_dirs[@]}"; do
  include_args+=(--include "$directory/*" --include "$directory/**")
done

echo "Uploading logs"
printf '  directories:'
printf ' %q' "${selected_dirs[@]}"
printf '\n'
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
