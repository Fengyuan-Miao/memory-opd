#!/usr/bin/env bash
# OPD-MM | KL-guided action-credit GRPO with all-fail privileged distillation

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS:-True}
WANDB_PROXY=${WANDB_PROXY:-}
WANDB_PROXY_FALLBACK=${WANDB_PROXY_FALLBACK:-http://127.0.0.1:7896}
WANDB_CONNECTIVITY_TIMEOUT=${WANDB_CONNECTIVITY_TIMEOUT:-5}
if [[ "${WANDB_MODE,,}" == "online" && -z "$WANDB_PROXY" ]] \
    && ! curl -sS --max-time "$WANDB_CONNECTIVITY_TIMEOUT" -o /dev/null https://api.wandb.ai; then
    if curl -sS --max-time "$WANDB_CONNECTIVITY_TIMEOUT" --proxy "$WANDB_PROXY_FALLBACK" \
        -o /dev/null https://api.wandb.ai; then
        WANDB_PROXY=$WANDB_PROXY_FALLBACK
    else
        echo "W&B is unreachable directly and through $WANDB_PROXY_FALLBACK" >&2
    fi
fi
export WANDB_MODE
WANDB_TRAINER_ARGS=(+trainer.wandb_disable_stats=${WANDB_DISABLE_STATS})
if [[ -n "$WANDB_PROXY" ]]; then
    WANDB_TRAINER_ARGS+=(+trainer.wandb_proxy="$WANDB_PROXY")
fi

# Start the student from the base 4B model by default. OPD_MODEL_PATH may still
# point to another merged HF checkpoint, or OPD_CHECKPOINT_DIR to a verl step.
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
    OPD_MODEL_PATH=/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B
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
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
TEACHER_GPU_MEMORY_UTIL=${TEACHER_GPU_MEMORY_UTIL:-0.55}
OPD_MM_KL_TOPK=${OPD_MM_KL_TOPK:-8}
OPD_MM_KL_TOP_ACTIONS=${OPD_MM_KL_TOP_ACTIONS:-2}

# One fixed VLM serves INSPECT_RAW, terminal answer generation, and the private
# correctness judge. Gold is sent only in the second, post-rollout judge call.
START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-1}
OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-9B}
OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-qwen35-9b-outcome}
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
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
actor_sp_size=${ACTOR_SP_SIZE:-4}
# Dynamic batching multiplies this per-device budget by the Ulysses SP size.
# Keep one complete 16K+2K state lossless while sharding its token/vocab
# activations across all four actor GPUs.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length + actor_sp_size - 1) / actor_sp_size))}
actor_use_torch_compile=${ACTOR_USE_TORCH_COMPILE:-False}
distill_chunk_size=${DISTILL_CHUNK_SIZE:-256}
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
experiment_name=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_selfdistill_klcredit_top2_grpo_${RUN_TIMESTAMP}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/${project_name}/${experiment_name}}
LOG_DIR=${LOG_DIR:-logs}
TRAIN_LOG_PATH=${TRAIN_LOG_PATH:-${LOG_DIR}/${experiment_name}.log}
OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR:-${LOG_DIR}/opd_mm_grpo_rollouts_${RUN_TIMESTAMP}}
OPD_MM_OUTCOME_REWARD_DUMP_DIR=${OPD_MM_OUTCOME_REWARD_DUMP_DIR:-${LOG_DIR}/opd_mm_grpo_outcomes_${RUN_TIMESTAMP}}
OPD_MM_TEACHER_CORRECTION_DUMP_DIR=${OPD_MM_TEACHER_CORRECTION_DUMP_DIR:-${LOG_DIR}/opd_mm_kl_credit_${RUN_TIMESTAMP}}
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

mkdir -p "$LOG_DIR" "$OPD_MM_STUDENT_ROLLOUT_DUMP_DIR" "$OPD_MM_OUTCOME_REWARD_DUMP_DIR" \
    "$OPD_MM_TEACHER_CORRECTION_DUMP_DIR" "$RAY_TMPDIR"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_TMPDIR TMPDIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_DIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS=${OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS:-12000}
export OPD_MM_RECORD_POLICY_STATES=1
export OPD_MM_KL_TOPK
export OPD_MM_KL_CREDIT_ASSIGNMENT=1
export OPD_MM_SKIP_INITIAL_CORRECTION=0
export OPD_MM_FAIL_ON_PROMPT_TRUNCATION=1
export OPD_MM_TEACHER_CORRECTION_DUMP_DIR
export OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS=${OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS:-12000}
export OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT=${OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT:-0}
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
echo "TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}"
echo "OPD_MM_KL_TOPK=${OPD_MM_KL_TOPK}"
echo "OPD_MM_KL_TOP_ACTIONS=${OPD_MM_KL_TOP_ACTIONS}"
echo "ACTOR_SP_SIZE=${actor_sp_size}"
echo "PPO_MAX_TOKEN_LEN_PER_GPU=${ppo_max_token_len_per_gpu}"
echo "DISTILL_CHUNK_SIZE=${distill_chunk_size}"
echo "ROLLOUT_N=${rollout_n}"
echo "TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
echo "OPD_MM_OUTCOME_REWARD_DUMP_DIR=${OPD_MM_OUTCOME_REWARD_DUMP_DIR}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROXY=${WANDB_PROXY:-direct}"
echo "WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS}"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    +algorithm.opd_mm_kl_credit.enabled=True
    +algorithm.opd_mm_kl_credit.top_actions=${OPD_MM_KL_TOP_ACTIONS}
    +algorithm.opd_mm_kl_credit.success_key=opd_mm/answer_correct
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
    actor_rollout_ref.actor.use_torch_compile=${actor_use_torch_compile}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${actor_sp_size}
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
    trainer.logger='["console","wandb"]'
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

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=${OPD_MM_KL_TOPK}
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.distillation_loss_coef=1.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    +distillation.distillation_loss.use_chunked_topk=True
    +distillation.distillation_loss.chunked_topk_chunk_size=${distill_chunk_size}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL_PATH"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS}
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
    distillation.teacher_models.teacher_model.inference.enforce_eager=False
)

set +e
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${WANDB_TRAINER_ARGS[@]}" \
    "${REWARD[@]}" \
    "${DISTILLATION[@]}" \
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
