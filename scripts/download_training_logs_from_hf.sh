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
LATEST_DIR_ONLY=${LATEST_DIR_ONLY:-1}
LOG_SUBDIR=${LOG_SUBDIR:-}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}

for command_name in hf python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing '$command_name'. Install huggingface_hub with: pip install -U huggingface_hub" >&2
    exit 1
  fi
done

token=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
token_args=()
if [[ -n "$token" ]]; then
  token_args=(--token "$token")
elif ! hf auth whoami >/dev/null 2>&1; then
  echo "No Hugging Face authentication found." >&2
  echo "Set HF_TOKEN or run: hf auth login" >&2
  exit 1
fi

include_args=()
selected_source="all logs"
if [[ -n "$LOG_SUBDIR" ]]; then
  if [[ "$LOG_SUBDIR" == /* || "$LOG_SUBDIR" == *".."* ]]; then
    echo "LOG_SUBDIR must be a relative directory without '..': $LOG_SUBDIR" >&2
    exit 1
  fi
  include_args=(--include "$LOG_SUBDIR/*" --include "$LOG_SUBDIR/**")
  selected_source=$LOG_SUBDIR
elif [[ "$LATEST_DIR_ONLY" != "0" && "$LATEST_DIR_ONLY" != "false" ]]; then
  latest_dir=$(
    HF_DOWNLOAD_TOKEN="$token" \
    HF_DOWNLOAD_REPO_ID="$HF_REPO_ID" \
    HF_DOWNLOAD_REPO_TYPE="$HF_REPO_TYPE" \
    HF_DOWNLOAD_REVISION="$HF_REVISION" \
    python3 - <<'PY'
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFolder

folders = [
    entry
    for entry in HfApi().list_repo_tree(
        repo_id=os.environ["HF_DOWNLOAD_REPO_ID"],
        repo_type=os.environ["HF_DOWNLOAD_REPO_TYPE"],
        revision=os.environ["HF_DOWNLOAD_REVISION"],
        token=os.environ.get("HF_DOWNLOAD_TOKEN") or None,
        expand=True,
    )
    if isinstance(entry, RepoFolder)
]
if not folders:
    sys.exit("No log directories found in the Hugging Face repository")

folders.sort(
    key=lambda entry: (
        entry.last_commit.date.timestamp() if entry.last_commit is not None else float("-inf"),
        entry.path,
    ),
    reverse=True,
)
print(folders[0].path)
PY
  )
  include_args=(--include "$latest_dir/*" --include "$latest_dir/**")
  selected_source=$latest_dir
fi

force_arg=--no-force-download
if [[ "$FORCE_DOWNLOAD" == "1" || "$FORCE_DOWNLOAD" == "true" ]]; then
  force_arg=--force-download
fi

mkdir -p "$LOGS_DIR"

echo "Downloading training logs"
echo "  source:   hf://datasets/$HF_REPO_ID ($selected_source)"
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
  "${include_args[@]}" \
  "${token_args[@]}"

echo "Download completed: $LOGS_DIR"
