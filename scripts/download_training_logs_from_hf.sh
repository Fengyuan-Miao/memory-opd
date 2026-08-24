#!/usr/bin/env bash
# Download OPD/MMem training logs from a Hugging Face dataset repository.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LOGS_DIR=${LOGS_DIR:-$REPO_ROOT/logs}
HF_REPO_ID=${HF_REPO_ID:-memory-rl/opd-mm-training-logs}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
HF_REVISION=${HF_REVISION:-main}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-8}
LOG_SUBDIR=${LOG_SUBDIR:-}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}

for command_name in hf python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing '$command_name'. Install huggingface_hub with: pip install -U huggingface_hub" >&2
    exit 1
  fi
done

mkdir -p "$LOGS_DIR"

remote_dir_output=$(
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
):
    if isinstance(entry, RepoFolder):
        print(entry.path)
PY
)
remote_dirs=()
if [[ -n "$remote_dir_output" ]]; then
  mapfile -t remote_dirs <<<"$remote_dir_output"
fi

selected_dirs=()
if [[ -n "$LOG_SUBDIR" ]]; then
  if [[ "$LOG_SUBDIR" == /* || "$LOG_SUBDIR" == *".."* || "$LOG_SUBDIR" == */* ]]; then
    echo "LOG_SUBDIR must be a single relative directory name: $LOG_SUBDIR" >&2
    exit 1
  fi
  remote_found=0
  for directory in "${remote_dirs[@]}"; do
    if [[ "$directory" == "$LOG_SUBDIR" ]]; then
      remote_found=1
      break
    fi
  done
  if (( ! remote_found )); then
    echo "Requested log directory does not exist on Hugging Face: $LOG_SUBDIR" >&2
    exit 1
  fi
  if [[ ! -d "$LOGS_DIR/$LOG_SUBDIR" ]]; then
    selected_dirs+=("$LOG_SUBDIR")
  fi
else
  for directory in "${remote_dirs[@]}"; do
    if [[ ! -d "$LOGS_DIR/$directory" ]]; then
      selected_dirs+=("$directory")
    fi
  done
fi

if (( ${#selected_dirs[@]} == 0 )); then
  echo "No new Hugging Face log directories to download."
  exit 0
fi

include_args=()
for directory in "${selected_dirs[@]}"; do
  include_args+=(--include "$directory/*" --include "$directory/**")
done

force_arg=--no-force-download
if [[ "$FORCE_DOWNLOAD" == "1" || "$FORCE_DOWNLOAD" == "true" ]]; then
  force_arg=--force-download
fi

echo "Downloading training logs"
echo "  source:   hf://datasets/$HF_REPO_ID"
printf '  directories:'
printf ' %q' "${selected_dirs[@]}"
printf '\n'
echo "  revision: $HF_REVISION"
echo "  target:   $LOGS_DIR"

# Existing files are reused by default, so interrupted downloads can resume
# without fetching completed artifacts again.
hf download "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --revision "$HF_REVISION" \
  --local-dir "$LOGS_DIR" \
  --max-workers "$HF_MAX_WORKERS" \
  "$force_arg" \
  "${include_args[@]}"

echo "Download completed: $LOGS_DIR"
