#!/usr/bin/env bash
# On-policy distillation | OPD-MM tool agent | vLLM rollout | FSDP training

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ---- user-adjustable ----
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
STUDENT_MODEL=${STUDENT_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}
# Frozen-teacher OPD-MM path. The same teacher vLLM service is also used by the
# verifier and INSPECT_RAW when OPD_MM_RAW_INSPECTOR_BACKEND=teacher.
TEACHER_MODEL=${TEACHER_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-9B}

OPD_MM_TRAIN_FILES=${OPD_MM_TRAIN_FILES:-"['/home/miaofy/memory-opd/dataset/mem_gallery/opd_mm_store/subsets/balanced_train_cap2/train.parquet']"}
OPD_MM_VAL_FILES=${OPD_MM_VAL_FILES:-$OPD_MM_TRAIN_FILES}
OPD_MM_TOOL_CONFIG=${OPD_MM_TOOL_CONFIG:-examples/opd_mm_baseline/opd_mm_tool_config.yaml}
OPD_MM_REWARD_PATH=${OPD_MM_REWARD_PATH:-/home/miaofy/memory-opd/verl/experimental/opd_mm/reward_manager.py}
OPD_MM_REWARD_NAME=${OPD_MM_REWARD_NAME:-compute_score}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
TEACHER_NGPUS=${TEACHER_NGPUS:-2}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
train_batch_size=${TRAIN_BATCH_SIZE:-$(( NGPUS_PER_NODE * per_gpu_batch_size ))}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
# Current OPD-MM system prompt plus OpenAI tool schemas is about 2.5k tokens
# before any tool observations, so keep the initial prompt budget above 2k.
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-16384}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.35}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:--1}

# Run the fixed 100-sample evidence-answerable evaluation after a successful
# training run. Set RUN_POST_TRAIN_EVAL=0 to keep training-only behavior.
RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL:-1}
POST_TRAIN_EVAL_GPUS=${POST_TRAIN_EVAL_GPUS:-0,1}
POST_TRAIN_EVAL_JUDGE_MODEL=${POST_TRAIN_EVAL_JUDGE_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3-VL-8B-Instruct}
POST_TRAIN_EVAL_OUTPUT_DIR=${POST_TRAIN_EVAL_OUTPUT_DIR:-outputs/opd_mm_eval}
POST_TRAIN_EVAL_DELAY_SECONDS=${POST_TRAIN_EVAL_DELAY_SECONDS:-15}
POST_TRAIN_EVAL_SEED=${POST_TRAIN_EVAL_SEED:-20260705}
POST_TRAIN_EVAL_MAX_TURNS=${POST_TRAIN_EVAL_MAX_TURNS:-8}
POST_TRAIN_EVAL_JUDGE_MAX_MODEL_LEN=${POST_TRAIN_EVAL_JUDGE_MAX_MODEL_LEN:-40000}

project_name=${PROJECT_NAME:-verl_distill_opd_mm}
experiment_name=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_teacher9b_verifierfeedback_sanitized_${RUN_TIMESTAMP}}
POST_TRAIN_CHECKPOINT_ROOT=${POST_TRAIN_CHECKPOINT_ROOT:-checkpoints/${project_name}/${experiment_name}}

LOG_DIR=${LOG_DIR:-logs}
TRAIN_LOG_PATH=${TRAIN_LOG_PATH:-${LOG_DIR}/${experiment_name}.log}
OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR:-${LOG_DIR}/opd_mm_student_rollouts_verifierfeedback_sanitized_${RUN_TIMESTAMP}}
OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS=${OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS:-12000}
OPD_MM_TEACHER_CORRECTION_DUMP_DIR=${OPD_MM_TEACHER_CORRECTION_DUMP_DIR:-${LOG_DIR}/opd_mm_teacher_corrections_verifierfeedback_sanitized_${RUN_TIMESTAMP}}
OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS=${OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS:-12000}
OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT=${OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT:-1}
OPD_MM_RAW_INSPECTOR_BACKEND=${OPD_MM_RAW_INSPECTOR_BACKEND:-teacher}
OPD_MM_RAW_INSPECTOR_MAX_TOKENS=${OPD_MM_RAW_INSPECTOR_MAX_TOKENS:-256}
OPD_MM_RAW_INSPECTOR_TEMPERATURE=${OPD_MM_RAW_INSPECTOR_TEMPERATURE:-0.0}
# Include the empty-pool (stage 0) state in online correction SFT data.
OPD_MM_SKIP_INITIAL_CORRECTION=${OPD_MM_SKIP_INITIAL_CORRECTION:-False}

# Ray creates Unix-domain sockets under its temp directory. Keep this path short:
# long paths can exceed the 107-byte AF_UNIX socket limit, while /tmp may be
# full on shared machines.
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/home/miaofy/rt}
RAY_TMPDIR=${RAY_TMPDIR:-${RAY_TMP_ROOT}/opd${RUN_TIMESTAMP:9}}
TMPDIR=${TMPDIR:-$RAY_TMPDIR}
# ---- end user-adjustable ----

mkdir -p "$LOG_DIR" "$OPD_MM_STUDENT_ROLLOUT_DUMP_DIR" "$OPD_MM_TEACHER_CORRECTION_DUMP_DIR" "$RAY_TMPDIR"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_TMPDIR TMPDIR
export OPD_MM_STUDENT_ROLLOUT_DUMP_DIR OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS
export OPD_MM_TEACHER_CORRECTION_DUMP_DIR OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS
export OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT
export OPD_MM_RAW_INSPECTOR_BACKEND
export OPD_MM_RAW_INSPECTOR_MAX_TOKENS OPD_MM_RAW_INSPECTOR_TEMPERATURE
export OPD_MM_SKIP_INITIAL_CORRECTION

exec > >(tee -a "$TRAIN_LOG_PATH") 2>&1
echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
echo "EXPERIMENT_NAME=${experiment_name}"
echo "TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR}"
echo "OPD_MM_TEACHER_CORRECTION_DUMP_DIR=${OPD_MM_TEACHER_CORRECTION_DUMP_DIR}"
echo "OPD_MM_RAW_INSPECTOR_BACKEND=${OPD_MM_RAW_INSPECTOR_BACKEND}"
echo "OPD_MM_SKIP_INITIAL_CORRECTION=${OPD_MM_SKIP_INITIAL_CORRECTION}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "TEACHER_MAX_MODEL_LEN=${teacher_max_model_len}"
echo "RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL}"
echo "POST_TRAIN_CHECKPOINT_ROOT=${POST_TRAIN_CHECKPOINT_ROOT}"
echo "POST_TRAIN_EVAL_GPUS=${POST_TRAIN_EVAL_GPUS}"
echo "POST_TRAIN_EVAL_JUDGE_MODEL=${POST_TRAIN_EVAL_JUDGE_MODEL}"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$OPD_MM_TRAIN_FILES"
    data.val_files="$OPD_MM_VAL_FILES"
    data.prompt_key=prompt
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
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
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
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
)

REWARD=(
    reward.custom_reward_function.path="$OPD_MM_REWARD_PATH"
    reward.custom_reward_function.name="$OPD_MM_REWARD_NAME"
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS}
    distillation.nnodes=${NNODES}
    +distillation.teacher_models.opd_mm.key=opd_mm
    +distillation.teacher_models.opd_mm.model_path="$TEACHER_MODEL"
    +distillation.teacher_models.opd_mm.num_replicas=1
    +distillation.teacher_models.opd_mm.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.opd_mm.inference.name=vllm
    +distillation.teacher_models.opd_mm.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.opd_mm.inference.max_model_len=${teacher_max_model_len}
    distillation.teacher_key=data_source
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

set +e
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@"
train_status=$?
set -e

if (( train_status != 0 )); then
    echo "Training failed with exit code ${train_status}; post-train evaluation is skipped."
    exit "$train_status"
fi

case "${RUN_POST_TRAIN_EVAL,,}" in
    1|true|yes|on)
        echo "Training completed; starting post-train OPD-MM evaluation."
        sleep "$POST_TRAIN_EVAL_DELAY_SECONDS"

        mapfile -t checkpoint_dirs < <(
            find "$POST_TRAIN_CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V
        )
        if (( ${#checkpoint_dirs[@]} == 0 )); then
            echo "No global_step checkpoint found under ${POST_TRAIN_CHECKPOINT_ROOT}" >&2
            exit 1
        fi
        final_checkpoint=${checkpoint_dirs[$((${#checkpoint_dirs[@]} - 1))]}
        final_step=$(basename "$final_checkpoint")
        prepared_model_dir="$final_checkpoint/actor_merged_hf_vllm_fixed"
        eval_output="$POST_TRAIN_EVAL_OUTPUT_DIR/${experiment_name}_${final_step}_100_evidenceanswerable_qwen3vl8b.jsonl"
        mkdir -p "$POST_TRAIN_EVAL_OUTPUT_DIR"

        echo "POST_TRAIN_FINAL_CHECKPOINT=${final_checkpoint}"
        echo "POST_TRAIN_PREPARED_MODEL=${prepared_model_dir}"
        echo "POST_TRAIN_EVAL_OUTPUT=${eval_output}"

        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 examples/opd_mm_baseline/prepare_opd_mm_checkpoint.py \
            --checkpoint-dir "$final_checkpoint" \
            --output-dir "$prepared_model_dir"

        CUDA_VISIBLE_DEVICES="$POST_TRAIN_EVAL_GPUS" \
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python3 examples/opd_mm_baseline/evaluate_opd_mm_llm_judge.py \
            --student-model "$prepared_model_dir" \
            --judge-model "$POST_TRAIN_EVAL_JUDGE_MODEL" \
            --judge-mode evidence_answerable \
            --output "$eval_output" \
            --max-samples 100 \
            --seed "$POST_TRAIN_EVAL_SEED" \
            --max-turns "$POST_TRAIN_EVAL_MAX_TURNS" \
            --student-tp 1 \
            --student-gpu-memory-utilization 0.35 \
            --max-model-len 8192 \
            --max-new-tokens 1024 \
            --temperature 0.0 \
            --judge-tp 2 \
            --judge-gpu-memory-utilization 0.55 \
            --judge-max-model-len "$POST_TRAIN_EVAL_JUDGE_MAX_MODEL_LEN" \
            --judge-max-new-tokens 192

        echo "Post-train evaluation summary:"
        cat "${eval_output%.jsonl}.summary.json"
        ;;
    0|false|no|off)
        echo "RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL}; post-train evaluation skipped."
        ;;
    *)
        echo "Invalid RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL}; expected 0/1 or false/true." >&2
        exit 2
        ;;
esac
