#!/usr/bin/env bash
# Mem-Gallery teacher-success subset | online per-state correction SFT.
#
# The student first visits states through its own tool-agent rollout. For each
# state, the verifier diagnoses the public evidence and the privileged teacher
# emits one canonical XML tool call. Cross-entropy is applied only to that
# target tool call; student rollout tokens and prompt tokens are not targets.
#
# Resource layout and portable dataset/model paths are inherited from
# run_opd_mm_memgallery_opsd.sh:
#   GPUs 0-2: 4B student and query encoders
#   GPU 3: frozen 4B correction teacher
#   GPUs 4-5: local 9B verifier / selector / answer / judge / image inspector

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export OPD_MM_ONLINE_SUPERVISION_MODE=correction_sft
export PROJECT_NAME=${PROJECT_NAME:-verl_sft_opd_mm_memgallery}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_memgallery_teacher_success1239_online_sft_$(date +%Y%m%d_%H%M%S)}

# Keep all paths relative to the checkout by default. The delegated launcher
# also accepts DATASET_ROOT, MMEM_DATASET_ROOT, MODEL_ROOT, STUDENT_MODEL, and
# TEACHER_MODEL overrides for mounted data/model directories.
export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
export DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT/dataset/mem_gallery}

exec bash "$SCRIPT_DIR/run_opd_mm_memgallery_opsd.sh" "$@"
