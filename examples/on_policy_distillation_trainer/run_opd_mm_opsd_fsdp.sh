#!/usr/bin/env bash
# OPD-MM | pure on-policy self-distillation on student-generated trajectories

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_DISABLE_STATS=${WANDB_DISABLE_STATS:-True}
WANDB_VAL_CASES=${WANDB_VAL_CASES:-4}
WANDB_VAL_ACTIONS_ONLY=${WANDB_VAL_ACTIONS_ONLY:-False}
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}
export WANDB_DISABLE_CODE=${WANDB_DISABLE_CODE:-true}
export WANDB_SAVE_CODE=${WANDB_SAVE_CODE:-false}
export WANDB_LOG_MODEL=${WANDB_LOG_MODEL:-false}
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
WANDB_TRAINER_ARGS=(
    +trainer.wandb_disable_stats=${WANDB_DISABLE_STATS}
    +trainer.wandb_val_actions_only=${WANDB_VAL_ACTIONS_ONLY}
    '+trainer.wandb_metric_include_patterns=["^(actor|critic|distillation)/.*loss$","^val-aux/.*/opd_mm/(answer_correct|evidence_answerable|evidence_count|action_count|repeated_actions|max_actions_reached|empty_evidence|trajectory_error|answer_request_failed|judge_request_failed|evidence_judge_request_failed)/mean@.*$","^training/(global_step|epoch)$"]'
)
if [[ -n "$WANDB_PROXY" ]]; then
    WANDB_TRAINER_ARGS+=(+trainer.wandb_proxy="$WANDB_PROXY")
fi

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-$STUDENT_MODEL}
export OPD_MM_DENSE_MODEL_PATH=${OPD_MM_DENSE_MODEL_PATH:-$MODEL_ROOT/all-MiniLM-L6-v2}
export OPD_MM_VISION_MODEL_PATH=${OPD_MM_VISION_MODEL_PATH:-$MODEL_ROOT/SigLIP-Base-Patch16-384}
export OPD_MM_HYBRID_MODEL_PATH=${OPD_MM_HYBRID_MODEL_PATH:-$MODEL_ROOT/gme-Qwen2-VL-2B-Instruct}
export OPD_MM_DATASET_ROOT=${OPD_MM_DATASET_ROOT:-$REPO_ROOT/dataset/mem_gallery}

DATA_DIR=${DATA_DIR:-$REPO_ROOT/dataset/mem_gallery/opd_mm_store/subsets/balanced_grpo_cap4_holdout100}
OPD_MM_TRAIN_FILES=${OPD_MM_TRAIN_FILES:-"['${DATA_DIR}/train.parquet']"}
OPD_MM_HELDOUT_QAS=${OPD_MM_HELDOUT_QAS:-${DATA_DIR}/heldout_qas.jsonl}
OPD_MM_VAL_PARQUET=${OPD_MM_VAL_PARQUET:-${DATA_DIR}/heldout_opsd_eval.parquet}
OPD_MM_VAL_FILES=${OPD_MM_VAL_FILES:-"['${OPD_MM_VAL_PARQUET}']"}
OPD_MM_TOOL_CONFIG=${OPD_MM_TOOL_CONFIG:-$REPO_ROOT/examples/opd_mm_baseline/opd_mm_tool_config.yaml}
OPD_MM_REWARD_PATH=${OPD_MM_REWARD_PATH:-$REPO_ROOT/verl/experimental/opd_mm/outcome_reward.py}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5}
OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS:-6,7}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
TEACHER_NNODES=${TEACHER_NNODES:-1}

START_OUTCOME_SERVER=${START_OUTCOME_SERVER:-1}
OUTCOME_MODEL_PATH=${OUTCOME_MODEL_PATH:-$STUDENT_MODEL}
OUTCOME_SERVED_MODEL=${OUTCOME_SERVED_MODEL:-qwen35-4b-opsd-eval}
OUTCOME_SERVER_HOST=${OUTCOME_SERVER_HOST:-127.0.0.1}
OUTCOME_SERVER_PORT=${OUTCOME_SERVER_PORT:-8011}
OUTCOME_SERVER_BASE_URL=${OUTCOME_SERVER_BASE_URL:-http://${OUTCOME_SERVER_HOST}:${OUTCOME_SERVER_PORT}}
OUTCOME_SERVER_TP=${OUTCOME_SERVER_TP:-2}
OUTCOME_SERVER_GPU_MEMORY_UTIL=${OUTCOME_SERVER_GPU_MEMORY_UTIL:-0.85}
OUTCOME_SERVER_MAX_MODEL_LEN=${OUTCOME_SERVER_MAX_MODEL_LEN:-40000}
OUTCOME_SERVER_MAX_NUM_SEQS=${OUTCOME_SERVER_MAX_NUM_SEQS:-64}
OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS=${OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS:-32768}
OUTCOME_SERVER_START_TIMEOUT=${OUTCOME_SERVER_START_TIMEOUT:-900}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
val_batch_size=${VAL_BATCH_SIZE:-20}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
actor_sp_size=${ACTOR_SP_SIZE:-4}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length + actor_sp_size - 1) / actor_sp_size))}
actor_lr=${ACTOR_LR:-5e-7}
actor_use_torch_compile=${ACTOR_USE_TORCH_COMPILE:-False}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.8}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}
val_rollout_temperature=${VAL_ROLLOUT_TEMPERATURE:-$rollout_temperature}
val_rollout_top_p=${VAL_ROLLOUT_TOP_P:-$rollout_top_p}
val_rollout_top_k=${VAL_ROLLOUT_TOP_K:--1}
val_rollout_do_sample=${VAL_ROLLOUT_DO_SAMPLE:-False}
val_before_train=${VAL_BEFORE_TRAIN:-True}

teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEMORY_UTIL:-0.55}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-40000}
teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
teacher_max_num_seqs=${TEACHER_MAX_NUM_SEQS:-16}
distillation_topk=${DISTILLATION_TOPK:-50}
distill_chunk_size=${DISTILL_CHUNK_SIZE:-256}

total_epochs=${TOTAL_EPOCHS:-3}
total_training_steps=${TOTAL_TRAINING_STEPS:-}
test_freq=${TEST_FREQ:-30}
# A positive frequency larger than the run makes ray_trainer save only at the
# mandatory final step.
save_freq=${SAVE_FREQ:-1000000}
reward_workers=${REWARD_WORKERS:-8}
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-8}

project_name=${PROJECT_NAME:-verl_opsd_opd_mm}
experiment_name=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_pure_opsd_current_schema_${RUN_TIMESTAMP}}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
TRAIN_LOG_PATH=${TRAIN_LOG_PATH:-${LOG_DIR}/${experiment_name}.log}
OUTCOME_SERVER_LOG=${OUTCOME_SERVER_LOG:-${LOG_DIR}/${experiment_name}_outcome_server.log}
OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR:-${LOG_DIR}/opd_mm_opsd_rollouts_${RUN_TIMESTAMP}}
OPD_MM_TEACHER_CORRECTION_DUMP_DIR=${OPD_MM_TEACHER_CORRECTION_DUMP_DIR:-${LOG_DIR}/opd_mm_opsd_corrections_${RUN_TIMESTAMP}}
OPD_MM_OUTCOME_REWARD_DUMP_DIR=${OPD_MM_OUTCOME_REWARD_DUMP_DIR:-${LOG_DIR}/opd_mm_opsd_validation_${RUN_TIMESTAMP}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/opd_mm_opsd_validation_generations_${RUN_TIMESTAMP}}

# Ray appends long session/socket suffixes, while AF_UNIX paths are limited to
# 107 bytes on Linux. Keep the default root independent of the checkout path.
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/verl_ray_$(id -u)}
RAY_TMPDIR=${RAY_TMPDIR:-${RAY_TMP_ROOT}/opsd${RUN_TIMESTAMP:9}}
TMPDIR=${TMPDIR:-$RAY_TMPDIR}

if [[ ! -s "$OPD_MM_VAL_PARQUET" || "$OPD_MM_HELDOUT_QAS" -nt "$OPD_MM_VAL_PARQUET" ]]; then
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 examples/data_preprocess/prepare_opd_mm_heldout_rlhf.py \
        --qas-jsonl "$OPD_MM_HELDOUT_QAS" \
        --output "$OPD_MM_VAL_PARQUET" \
        --data-source opd_mm_eval
fi

mkdir -p "$LOG_DIR" "$OPD_MM_STUDENT_ROLLOUT_DUMP_DIR" "$OPD_MM_TEACHER_CORRECTION_DUMP_DIR" \
    "$OPD_MM_OUTCOME_REWARD_DUMP_DIR" "$VALIDATION_DATA_DIR" "$RAY_TMPDIR"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_TMPDIR TMPDIR
export OPD_MM_ONLINE_SUPERVISION_MODE=opsd
export OPD_MM_RECORD_POLICY_STATES=1
export OPD_MM_KL_CREDIT_ASSIGNMENT=1
export OPD_MM_KL_TOPK="$distillation_topk"
export OPD_MM_FAIL_ON_PROMPT_TRUNCATION=1
export OPD_MM_STUDENT_ROLLOUT_DUMP_DIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS=${OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS:-12000}
export OPD_MM_TEACHER_CORRECTION_DUMP_DIR
export OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS=${OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS:-12000}
export OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT=${OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT:-0}
export OPD_MM_OUTCOME_REWARD_DUMP_DIR
export OPD_MM_OUTCOME_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_OUTCOME_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_JUDGE_BASE_URL="$OUTCOME_SERVER_BASE_URL"
export OPD_MM_JUDGE_MODEL="$OUTCOME_SERVED_MODEL"
export OPD_MM_VERIFIER_BASE_URL=${OPD_MM_VERIFIER_BASE_URL:-$OUTCOME_SERVER_BASE_URL}
export OPD_MM_VERIFIER_MODEL=${OPD_MM_VERIFIER_MODEL:-Qwen3.5-9B}
export OPD_MM_VERIFIER_TIMEOUT=${OPD_MM_VERIFIER_TIMEOUT:-120}
export OPD_MM_VERIFIER_RETRIES=${OPD_MM_VERIFIER_RETRIES:-3}
export OPD_MM_EVIDENCE_SELECTOR_BASE_URL=${OPD_MM_EVIDENCE_SELECTOR_BASE_URL:-$OPD_MM_VERIFIER_BASE_URL}
export OPD_MM_EVIDENCE_SELECTOR_MODEL=${OPD_MM_EVIDENCE_SELECTOR_MODEL:-$OPD_MM_VERIFIER_MODEL}
export OPD_MM_EVIDENCE_SELECTOR_BACKEND=${OPD_MM_EVIDENCE_SELECTOR_BACKEND:-remote}
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
            --max-num-seqs "$OUTCOME_SERVER_MAX_NUM_SEQS" \
            --max-num-batched-tokens "$OUTCOME_SERVER_MAX_NUM_BATCHED_TOKENS" \
            --enable-chunked-prefill \
            --disable-custom-all-reduce \
            --limit-mm-per-prompt '{"image":2}' \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
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

# Exercise the exact INSPECT_RAW path before Ray starts. RemoteVLLMRawInspector
# reads the complete local file and sends it as a data:<mime>;base64 URL, so
# this catches remote multimodal worker/proxy failures that /v1/models cannot.
if [[ -n "${OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE:-}" ]]; then
    if [[ ! -s "$OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE" ]]; then
        echo "Missing INSPECT_RAW health-check image: $OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE" >&2
        exit 1
    fi
    raw_inspector_health=$(
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - "$OPD_MM_RAW_INSPECTOR_HEALTHCHECK_IMAGE" <<'PY'
import os
import sys

from verl.experimental.opd_mm.raw_inspector import RemoteVLLMRawInspector

result = RemoteVLLMRawInspector(
    base_url=os.environ["OPD_MM_RAW_INSPECTOR_URL"],
    model=os.environ["OPD_MM_RAW_INSPECTOR_MODEL"],
    api_key=os.getenv("OPD_MM_RAW_INSPECTOR_API_KEY"),
    timeout=float(os.getenv("OPD_MM_RAW_INSPECTOR_TIMEOUT", "120")),
    max_tokens=32,
    temperature=0.0,
).inspect(sys.argv[1], "Identify the main visible subject in this image.")
if result.startswith("RAW_INSPECT_ERROR:"):
    print(result, file=sys.stderr)
    raise SystemExit(1)
print(result.replace("\n", " ")[:240])
PY
    ) || {
        echo "External INSPECT_RAW base64 health check failed; training was not started." >&2
        exit 1
    }
    echo "INSPECT_RAW_BASE64_HEALTHCHECK=${raw_inspector_health}"
fi

verifier_models_url=${OPD_MM_VERIFIER_BASE_URL%/}
if [[ "$verifier_models_url" != */v1 ]]; then
    verifier_models_url="$verifier_models_url/v1"
fi
curl --noproxy '*' -fsS --max-time 10 "$verifier_models_url/models" >/dev/null || {
    echo "External OPD-MM verifier is unavailable: $verifier_models_url" >&2
    exit 1
}

exec > >(tee -a "$TRAIN_LOG_PATH") 2>&1
echo "EXPERIMENT_NAME=${experiment_name}"
echo "STUDENT_MODEL=${STUDENT_MODEL}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "OPD_MM_VERIFIER_MODEL=${OPD_MM_VERIFIER_MODEL}"
echo "OPD_MM_VERIFIER_BASE_URL=${OPD_MM_VERIFIER_BASE_URL}"
echo "OPD_MM_EVIDENCE_SELECTOR_BASE_URL=${OPD_MM_EVIDENCE_SELECTOR_BASE_URL}"
echo "OPD_MM_EVIDENCE_SELECTOR_MODEL=${OPD_MM_EVIDENCE_SELECTOR_MODEL}"
echo "TRAIN_GPUS=${TRAIN_GPUS}"
echo "OUTCOME_SERVER_GPUS=${OUTCOME_SERVER_GPUS}"
echo "TRAIN_SAMPLES=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${DATA_DIR}/train.parquet').num_rows)")"
echo "VAL_SAMPLES=$(python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('${OPD_MM_VAL_PARQUET}').num_rows)")"
echo "TRAIN_BATCH_SIZE=${train_batch_size}"
echo "VAL_BATCH_SIZE=${val_batch_size}"
echo "TEST_FREQ=${test_freq}"
echo "SAVE_FREQ=${save_freq}"
echo "DISTILLATION_TOPK=${distillation_topk}"
echo "TEACHER_GPU_MEMORY_UTIL=${teacher_gpu_mem_util}"
echo "TEACHER_MAX_NUM_BATCHED_TOKENS=${teacher_max_num_batched_tokens}"
echo "TEACHER_MAX_NUM_SEQS=${teacher_max_num_seqs}"
echo "AGENT_LOOP_NUM_WORKERS=${agent_loop_num_workers}"
echo "VERL_AGENT_LOOP_WORKER_CUDA_DEVICES=${VERL_AGENT_LOOP_WORKER_CUDA_DEVICES:-disabled}"
echo "ROLLOUT_GPU_MEM_UTIL=${rollout_gpu_mem_util}"
echo "ROLLOUT_MAX_NUM_BATCHED_TOKENS=${rollout_max_num_batched_tokens}"
echo "VAL_ROLLOUT_TEMPERATURE=${val_rollout_temperature}"
echo "VAL_ROLLOUT_TOP_P=${val_rollout_top_p}"
echo "VAL_ROLLOUT_TOP_K=${val_rollout_top_k}"
echo "VAL_ROLLOUT_DO_SAMPLE=${val_rollout_do_sample}"
echo "VAL_BEFORE_TRAIN=${val_before_train}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROXY=${WANDB_PROXY:-direct}"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    +algorithm.opd_mm_state_opsd.enabled=True
    +algorithm.opd_mm_state_opsd.topk=${distillation_topk}
    data.train_files="$OPD_MM_TRAIN_FILES"
    data.val_files="$OPD_MM_VAL_FILES"
    data.prompt_key=prompt
    data.train_batch_size=${train_batch_size}
    data.val_batch_size=${val_batch_size}
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
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=${actor_use_torch_compile}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${actor_sp_size}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=${distillation_topk}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$OPD_MM_TOOL_CONFIG"
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=${agent_loop_num_workers}
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_rollout_do_sample}
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_rollout_temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_rollout_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_rollout_top_k}
)

TRAINER=(
    trainer.use_v1=False
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.log_val_generations=${WANDB_VAL_CASES}
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${val_before_train}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.resume_mode=disable
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=0
    trainer.validation_data_dir="$VALIDATION_DATA_DIR"
)
if [[ -n "$total_training_steps" ]]; then
    TRAINER+=(trainer.total_training_steps=${total_training_steps})
fi

REWARD=(
    reward.num_workers=${reward_workers}
    reward.custom_reward_function.path="$OPD_MM_REWARD_PATH"
    reward.custom_reward_function.name=compute_opsd_validation_score
    +reward.custom_reward_function.reward_kwargs.repeat_penalty=0.0
    +reward.custom_reward_function.reward_kwargs.max_action_penalty=0.0
    +reward.custom_reward_function.reward_kwargs.error_penalty=0.0
    +reward.custom_reward_function.reward_kwargs.non_stop_penalty=0.0
    +reward.custom_reward_function.reward_kwargs.empty_evidence_penalty=0.0
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    +distillation.distillation_loss.use_chunked_topk=True
    +distillation.distillation_loss.chunked_topk_chunk_size=${distill_chunk_size}
    distillation.teacher_models.teacher_model.key=opd_mm
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens}
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${teacher_max_num_seqs}
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
exit "$train_status"
