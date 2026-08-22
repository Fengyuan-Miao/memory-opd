#!/usr/bin/env bash
# Download all models required by the portable OPD-MM Validated21 launcher.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}

download_hf_model() {
    local repo_id=$1
    local target_dir=$2

    if [[ -f "$target_dir/config.json" ]]; then
        echo "Using cached model: $target_dir"
        return
    fi

    echo "Downloading $repo_id to $target_dir"
    python3 - "$repo_id" "$target_dir" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    endpoint="https://huggingface.co",
)
PY
}

mkdir -p "$MODEL_ROOT"

download_hf_model "${MODEL_4B_REPO:-Qwen/Qwen3.5-4B}" \
    "${MODEL_4B_PATH:-$MODEL_ROOT/Qwen3.5-4B}"
download_hf_model "${MODEL_9B_REPO:-Qwen/Qwen3.5-9B}" \
    "${MODEL_9B_PATH:-$MODEL_ROOT/Qwen3.5-9B}"
download_hf_model "${DENSE_MODEL_REPO:-sentence-transformers/all-MiniLM-L6-v2}" \
    "${OPD_MM_DENSE_MODEL_PATH:-$MODEL_ROOT/all-MiniLM-L6-v2}"
download_hf_model "${VISION_MODEL_REPO:-google/siglip2-base-patch16-384}" \
    "${OPD_MM_VISION_MODEL_PATH:-$MODEL_ROOT/SigLIP-Base-Patch16-384}"
download_hf_model "${HYBRID_MODEL_REPO:-Alibaba-NLP/gme-Qwen2-VL-2B-Instruct}" \
    "${OPD_MM_HYBRID_MODEL_PATH:-$MODEL_ROOT/gme-Qwen2-VL-2B-Instruct}"

echo "All OPD-MM models are available under: $MODEL_ROOT"
