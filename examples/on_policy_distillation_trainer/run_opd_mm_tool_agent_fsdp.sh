#!/usr/bin/env bash
# On-policy distillation | OPD-MM tool agent | vLLM rollout | FSDP training

set -xeuo pipefail

# ---- user-adjustable ----
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
STUDENT_MODEL=${STUDENT_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}
# Frozen-teacher OPD-MM path. The same teacher vLLM service is also used by the
# verifier and INSPECT_RAW when OPD_MM_RAW_INSPECTOR_BACKEND=teacher.
TEACHER_MODEL=${TEACHER_MODEL:-/home/guojr/data/pretrained_models/Qwen/Qwen3.5-4B}

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
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
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

project_name=${PROJECT_NAME:-verl_distill_opd_mm}
experiment_name=${EXPERIMENT_NAME:-opd_mm_qwen35_4b_teacher4b_skipstage0_verifierfeedback_sanitized_${RUN_TIMESTAMP}}

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

exec > >(tee -a "$TRAIN_LOG_PATH") 2>&1
echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
echo "EXPERIMENT_NAME=${experiment_name}"
echo "TRAIN_LOG_PATH=${TRAIN_LOG_PATH}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "OPD_MM_STUDENT_ROLLOUT_DUMP_DIR=${OPD_MM_STUDENT_ROLLOUT_DUMP_DIR}"
echo "OPD_MM_TEACHER_CORRECTION_DUMP_DIR=${OPD_MM_TEACHER_CORRECTION_DUMP_DIR}"
echo "OPD_MM_RAW_INSPECTOR_BACKEND=${OPD_MM_RAW_INSPECTOR_BACKEND}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "TEACHER_MAX_MODEL_LEN=${teacher_max_model_len}"

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

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@"
