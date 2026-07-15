#!/usr/bin/env bash
# OPD-MM stage 2 | answer-outcome GRPO from an OPD warm-start checkpoint

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

# Use OPD_MODEL_PATH for an already merged HF checkpoint. Alternatively set
# OPD_CHECKPOINT_DIR to a verl global_step_* directory and this script will
# validate/merge it before starting a fresh optimizer state.
OPD_MODEL_PATH=${OPD_MODEL_PATH:-}
OPD_CHECKPOINT_DIR=${OPD_CHECKPOINT_DIR:-}
if [[ -z "$OPD_MODEL_PATH" && -n "$OPD_CHECKPOINT_DIR" ]]; then
    OPD_MODEL_PATH="$OPD_CHECKPOINT_DIR/actor_merged_hf_vllm_fixed"
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 examples/opd_mm_baseline/prepare_opd_mm_checkpoint.py \
        --checkpoint-dir "$OPD_CHECKPOINT_DIR" \
        --output-dir "$OPD_MODEL_PATH"
fi
if [[ -z "$OPD_MODEL_PATH" ]]; then
    OPD_MODEL_PATH=$(find checkpoints/verl_distill_opd_mm -type d -name actor_merged_hf_vllm_fixed \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
fi
if [[ -z "$OPD_MODEL_PATH" || ! -f "$OPD_MODEL_PATH/config.json" ]]; then
    echo "No prepared OPD model found. Set OPD_MODEL_PATH or OPD_CHECKPOINT_DIR." >&2
    exit 1
fi
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$OPD_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

from examples.opd_mm_baseline.prepare_opd_mm_checkpoint import validate_vllm_checkpoint

print(json.dumps(validate_vllm_checkpoint(Path(sys.argv[1])), ensure_ascii=False, sort_keys=True))
PY

GRPO_DATA_DIR=${GRPO_DATA_DIR:-/home/miaofy/memory-opd/dataset/mem_gallery/opd_mm_store/subsets/balanced_grpo_cap4_holdout100}
OPD_MM_TRAIN_FILES=${OPD_MM_TRAIN_FILES:-"['${GRPO_DATA_DIR}/train.parquet']"}
OPD_MM_VAL_FILES=${OPD_MM_VAL_FILES:-$OPD_MM_TRAIN_FILES}
OPD_MM_TOOL_CONFIG=${OPD_MM_TOOL_CONFIG:-examples/opd_mm_baseline/opd_mm_tool_config.yaml}
OPD_MM_REWARD_PATH=${OPD_MM_REWARD_PATH:-/home/miaofy/memory-opd/verl/experimental/opd_mm/outcome_reward.py}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5}
OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-6,7}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}

# One fixed VLM serves INSPECT_RAW, terminal answer generation, and the private
# correctness judge. Gold is sent only in the second, post-rollout judge call.
START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-1}
OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH:-/home/guojr/data/pretrained_models/Qwen/Qwen3-VL-8B-Instruct}
OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-opd-mm-outcome}
OUTCOME_SERVER_HOST=${OUTCOME_SERVER_HOST:-127.0.0.1}
OUTCOME_SERVER_PORT=${OUTCOME_SERVER_PORT:-8011}
OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://${OUTCOME_SERVER_HOST}:${OUTCOME_SERVER_PORT}}
OUTCOME_SERVER_TP=${OUTCOME_SERVER_TP:-2}
OUTCOME_SERVER_GPU_MEMORY_UTIL=${OUTCOME_SERVER_GPU_MEMORY_UTIL:-0.9}
OUTCOME_SERVER_MAX_MODEL_LEN=${OUTCOME_SERVER_MAX_MODEL_LEN:-40000}
OUTCOME_SERVER_START_TIMEOUT=${OUTCOME_SERVER_START_TIMEOUT:-900}

rollout_n=${ROLLOUT_N:-4}
train_batch_size=${TRAIN_BATCH_SIZE:-12}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
actor_lr=${ACTOR_LR:-5e-7}
entropy_coeff=${ENTROPY_COEFF:-0.0}
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.8}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}
total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-25}
test_freq=${TEST_FREQ:--1}
reward_workers=${REWARD_WORKERS:-8}

REPEAT_PENALTY=${REPEAT_PENALTY:-0.02}
MAX_ACTION_PENALTY=${MAX_ACTION_PENALTY:-0.1}
ERROR_PENALTY=${ERROR_PENALTY:-0.1}
NON_STOP_PENALTY=${NON_STOP_PENALTY:-0.1}
EMPTY_EVIDENCE_PENALTY=${EMPTY_EVIDENCE_PENALTY:-0.1}

project_name=${PROJECT_NAME:-verl_grpo_opd_mm}
experiment_name=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_opd_warmstart_outcome_grpo_${RUN_TIMESTAMP}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/${project_name}/${experiment_name}}
LOG_DIR=${LOG_DIR:-logs}
TRAIN_LOG_PATH=${TRAIN_LOG_PATH:-${LOG_DIR}/${experiment_name}.log}
OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR:-${LOG_DIR}/opd_mm_grpo_rollouts_${RUN_TIMESTAMP}}
OPD_MM_OUTCOME_REWARD_DUMP_DIR=${OPD_MM_OUTCOME_REWARD_DUMP_DIR:-${LOG_DIR}/opd_mm_grpo_outcomes_${RUN_TIMESTAMP}}
OUTCOME_SERVER_LOG=${OUTCOME_SERVER_LOG:-${LOG_DIR}/${experiment_name}_outcome_server.log}

RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL:-1}
POST_TRAIN_EVAL_GPUS=${POST_TRAIN_EVAL_GPUS:-0,1}
POST_TRAIN_EVAL_OUTPUT_DIR=${POST_TRAIN_EVAL_OUTPUT_DIR:-outputs/opd_mm_eval}
POST_TRAIN_EVAL_SEED=${POST_TRAIN_EVAL_SEED:-20260705}
POST_TRAIN_EVAL_SAMPLE_IDS=${POST_TRAIN_EVAL_SAMPLE_IDS:-${GRPO_DATA_DIR}/heldout_sample_ids.txt}
POST_TRAIN_EVAL_RLHF_PATH=${POST_TRAIN_EVAL_RLHF_PATH:-${GRPO_DATA_DIR}/train.parquet}

RAY_TMP_ROOT=${RAY_TMP_ROOT:-/home/miaofy/rt}
RAY_TMPDIR=${RAY_TMPDIR:-${RAY_TMP_ROOT}/grpo${RUN_TIMESTAMP:9}}
TMPDIR=${TMPDIR:-$RAY_TMPDIR}

mkdir -p "$LOG_DIR" "$OPD_MM_STUDENT_ROLLOUT_DUMP_DIR" "$OPD_MM_OUTCOME_REWARD_DUMP_DIR" "$RAY_TMPDIR"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_TMPDIR TMPDIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_DIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS=${OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS:-12000}
export OPD_MM_RECORD_POLICY_STATES=1
export OPD_MM_OUTCOME_REWARD_DUMP_DIR
export OPD_MM_OUTCOME_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_OUTCOME_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_JUDGE_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_JUDGE_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_RAW_INSPECTOR_BACKEND=vllm
export OPD_MM_RAW_INSPECTOR_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_RAW_INSPECTOR_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_RAW_INSPECTOR_MAX_TOKENS=${OPD_MM_RAW_INSPECTOR_MAX_TOKENS:-256}
export OPD_MM_RAW_INSPECTOR_TEMPERATURE=0.0

outcome_server_pid=""
cleanup() {
    if [[ -n "$outcome_server_pid" ]] && kill -0 "$outcome_server_pid" 2>/dev/null; then
        kill "$outcome_server_pid" 2>/dev/null || true
        wait "$outcome_server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

case "${START_OUTCOME_SERVER,,}" in
    1|true|yes|on)
        CUDA_VISIBLE_DEVICES="$OUTCOME_SERVER_GPUS" \
            python3 -m vllm.entrypoints.openai.api_server \
            --model "$OUTCOME_MODEL_PATH" \
            --served-model-name "$OUTCOME_SERVED_MODEL" \
            --host "$OUTCOME_SERVER_HOST" \
            --port "$OUTCOME_SERVER_PORT" \
            --tensor-parallel-size "$OUTCOME_SERVER_TP" \
            --gpu-memory-utilization "$OUTCOME_SERVER_GPU_MEMORY_UTIL" \
            --max-model-len "$OUTCOME_SERVER_MAX_MODEL_LEN" \
            --trust-remote-code \
            >"$OUTCOME_SERVER_LOG" 2>&1 &
        outcome_server_pid=$!
        deadline=$((SECONDS + OUTCOME_SERVER_START_TIMEOUT))
        until curl -fsS "$OUTCOME_SERVER_BASE_URL/v1/models" >/dev/null 2>&1; do
            if ! kill -0 "$outcome_server_pid" 2>/dev/null; then
                echo "Outcome vLLM service exited during startup. See $OUTCOME_SERVER_LOG" >&2
                exit 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for outcome vLLM service. See $OUTCOME_SERVER_LOG" >&2
                exit 1
            fi
            sleep 5
        done
        ;;
    0|false|no|off)
        curl -fsS "$OUTCOME_SERVER_BASE_URL/v1/models" >/dev/null
        ;;
    *)
        echo "Invalid START_OUTCOME_SERVER=$START_OUTCOME_SERVER" >&2
        exit 1
        ;;
esac

exec > >(tee -a "$TRAIN_LOG_PATH") 2>&1
echo "OPD_MODEL_PATH=${OPD_MODEL_PATH}"
echo "EXPERIMENT_NAME=${experiment_name}"
echo "TRAIN_GPUS=${TRAIN_GPUS}"
echo "OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL}"
echo "OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH}"
echo "ROLLOUT_N=${rollout_n}"
echo "TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
echo "OPD_MM_OUTCOME_REWARD_DUMP_DIR=${OPD_MM_OUTCOME_REWARD_DUMP_DIR}"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=True
    algorithm.rollout_correction.loss_type=ppo_clip
    data.train_files="$OPD_MM_TRAIN_FILES"
    data.val_files="$OPD_MM_VAL_FILES"
    data.prompt_key=prompt
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.tool_config_path="$OPD_MM_TOOL_CONFIG"
    data.trust_remote_code=True
    data.continuous_token.enable=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="$OPD_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$OPD_MM_TOOL_CONFIG"
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=8
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.load_format=dummy
)

TRAINER=(
    trainer.use_v1=False
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.resume_mode=disable
)

REWARD=(
    reward.num_workers=${reward_workers}
    reward.custom_reward_function.path="$OPD_MM_REWARD_PATH"
    reward.custom_reward_function.name=compute_outcome_score
    +reward.custom_reward_function.reward_kwargs.repeat_penalty=${REPEAT_PENALTY}
    +reward.custom_reward_function.reward_kwargs.max_action_penalty=${MAX_ACTION_PENALTY}
    +reward.custom_reward_function.reward_kwargs.error_penalty=${ERROR_PENALTY}
    +reward.custom_reward_function.reward_kwargs.non_stop_penalty=${NON_STOP_PENALTY}
    +reward.custom_reward_function.reward_kwargs.empty_evidence_penalty=${EMPTY_EVIDENCE_PENALTY}
)

set +e
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    distillation.enabled=False \
    "$@"
train_status=$?
set -e
if (( train_status != 0 )); then
    exit "$train_status"
fi

case "${RUN_POST_TRAIN_EVAL,,}" in
    1|true|yes|on)
        mapfile -t checkpoint_dirs < <(
            find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V
        )
        if (( ${#checkpoint_dirs[@]} == 0 )); then
            echo "No GRPO checkpoint found under $CHECKPOINT_ROOT" >&2
            exit 1
        fi
        final_checkpoint=${checkpoint_dirs[$((${#checkpoint_dirs[@]} - 1))]}
        final_step=$(basename "$final_checkpoint")
        prepared_model_dir="$final_checkpoint/actor_merged_hf_vllm_fixed"
        eval_output="$POST_TRAIN_EVAL_OUTPUT_DIR/${experiment_name}_${final_step}_100_answercorrectness.jsonl"
        mkdir -p "$POST_TRAIN_EVAL_OUTPUT_DIR"
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 examples/opd_mm_baseline/prepare_opd_mm_checkpoint.py \
            --checkpoint-dir "$final_checkpoint" \
            --output-dir "$prepared_model_dir"
        CUDA_VISIBLE_DEVICES="$POST_TRAIN_EVAL_GPUS" \
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 examples/opd_mm_baseline/evaluate_opd_mm_llm_judge.py \
            --student-model "$prepared_model_dir" \
            --eval-sample-ids "$POST_TRAIN_EVAL_SAMPLE_IDS" \
            --train-rlhf-path "$POST_TRAIN_EVAL_RLHF_PATH" \
            --answer-base-url "$OUTCOME_SERVER_BASE_URL" \
            --answer-model "$OUTCOME_SERVED_MODEL" \
            --answer-max-evidence 100000 \
            --judge-model "$OUTCOME_MODEL_PATH" \
            --judge-mode answer_correctness \
            --output "$eval_output" \
            --max-samples 100 \
            --seed "$POST_TRAIN_EVAL_SEED" \
            --max-turns 10 \
            --student-tp 1 \
            --student-gpu-memory-utilization 0.35 \
            --max-model-len 8192 \
            --max-new-tokens 1024 \
            --temperature 0.0 \
            --judge-tp 2 \
            --judge-gpu-memory-utilization 0.55 \
            --judge-max-model-len "$OUTCOME_SERVER_MAX_MODEL_LEN"
        cat "${eval_output%.jsonl}.summary.json"
        ;;
    0|false|no|off)
        ;;
    *)
        echo "Invalid RUN_POST_TRAIN_EVAL=$RUN_POST_TRAIN_EVAL" >&2
        exit 1
        ;;
esac
