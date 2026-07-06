# OPD-MM baseline in verl

This example hosts the migrated OPD-MM hidden-memory retrieval environment for
verl's on-policy distillation pipeline. It is intentionally split into two
layers:

- verl.experimental.opd_mm: CPU-only environment, validator, retriever,
  executor, verl-native tools, SFT conversion helpers, and reward bridge.
- examples/opd_mm_baseline: wiring examples for verl configs and smoke runs.

The migration keeps the original OPD-MM isolation contract:

- the policy sees the user query and public tool schemas;
- the hidden memory store is available only to the OPD-MM tools/executor;
- tool observations omit memory IDs;
- retrieve uses the original user query by default and may optionally receive a
  rewritten query string for that retrieval step; hidden memory IDs are still
  rejected through the validator.

## verl-native tool path

OPD-MM uses verl ToolAgentLoop as the student rollout environment. Use
opd_mm_tool_config.yaml as the rollout multi-turn tool config:

~~~bash
actor_rollout_ref.rollout.multi_turn.enable=true \
actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/opd_mm_baseline/opd_mm_tool_config.yaml \
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
~~~

Per sample, pass hidden memory state through the non-tensor field tools_kwargs,
for example:

~~~python
{
    "raw_prompt": [
        {"role": "system", "content": "<OPD-MM retrieval tool-use instructions>"},
        {"role": "user", "content": "Which pet image did I upload last?"},
    ],
    "agent_name": "tool_agent",
    "tools_kwargs": {
        "opd_mm": {
            "query": "Which pet image did I upload last?",
            "records": [
                {
                    "memory_id": "m1",
                    "turn_id": "1",
                    "timestamp": "2026-01-01T10:00:00",
                    "author": "user",
                    "modality": "image",
                    "source_type": "uploaded_image",
                    "summary": "A tabby cat on a sofa.",
                    "raw_pointer": "images/cat.png",
                }
            ],
        }
    },
}
~~~

The public tool response contains pool/evidence counts and ID-free evidence.
The full hidden state remains inside the tool session.

## On-policy distillation data path

The main OPD-MM path should run through verl.trainer.main_ppo with
distillation.enabled=True. Convert OPD-MM samples to RLHF rows with:

~~~python
from verl.experimental.opd_mm.dataset import write_opd_rlhf_parquet

write_opd_rlhf_parquet(samples, "opd_mm_train.parquet")
~~~

Each row contains:

- data_source=opd_mm for teacher routing;
- agent_name=tool_agent so AgentLoopWorker uses ToolAgentLoop;
- prompt containing an OPD-MM system prompt plus the user query;
- extra_info.tools_kwargs.opd_mm containing hidden memory records for the
  OPD-MM tools.

The default OPD-MM system prompt introduces the retrieval tools and their use
cases: RETRIEVE for semantic search, FILTER for metadata constraints, SORT for
recency/ordering, TOPK for narrowing candidates, INSPECT_RAW for limited raw
visual/detail inspection, and STOP when enough evidence has been collected.
It does not include hidden memory records or the gold answer.

The training flow is:

~~~text
RLHFDataset row
  -> ToolAgentLoop student rollout with OPD-MM tools
  -> AgentLoopWorker builds an OPD-MM teacher-only privileged prompt
  -> AgentLoopWorker requests teacher logprobs for the on-policy response
  -> distillation loss trains the student on its own rollout distribution
~~~

Rows produced by write_opd_rlhf_parquet set
extra_info.teacher_privilege_mode=opd_mm. With that flag, teacher scoring is
conditioned on:

- the user question;
- the gold answer;
- the student's OPD-MM tool-call history and tool results from tool extra
  fields.

Those privileged fields are used only for teacher logprob scoring. The student
prompt remains the original memory-free query and tool schema.

The teacher config key should match the row data_source unless
distillation.teacher_key is overridden:

~~~bash
distillation.enabled=True \
+distillation.teacher_models.opd_mm.key=opd_mm \
+distillation.teacher_models.opd_mm.model_path="$TEACHER_MODEL" \
+distillation.teacher_models.opd_mm.num_replicas=1
~~~

## Original OPD-MM correction loop

The original OPD-MM closed loop is also available as a CPU-side correction
or data-generation component:

~~~python
from verl.experimental.opd_mm.on_policy_distiller import OnPolicyDistiller

distiller = OnPolicyDistiller(
    student=student_policy,
    teacher=teacher_policy,
    executor=tool_executor,
    answer_model=answer_model,
    judge=answer_judge,
)
rollouts = distiller.run_round(samples)
~~~

That flow is:

~~~text
student.generate_trace(query)
  -> ToolExecutor runs the on-policy trace on hidden memory
  -> answer_model answers from retrieved evidence
  -> judge verifies the answer against gold
  -> teacher.correct receives the failed/successful rollout context
  -> corrected teacher trajectory is replayed and selected
  -> SFTExample target stores the corrected action trace
~~~

Use this path when you want OPD-MM hindsight correction targets rather than
teacher logprobs on the student sequence. The two paths can be combined by
using corrected examples for warm start and verl's teacher-logprob OPD for
online training.

## Step-level correction collector

For multi-step retrieval, the strongest OPD-MM signal is the state-level target:

~~~text
query + previous actions + latest public observation -> corrected next action
~~~

Use StepCorrectionCollector to replay student-visited states and ask a teacher
for the corrected next action at each state:

~~~python
from verl.experimental.opd_mm.step_correction import StepCorrectionCollector

collector = StepCorrectionCollector(
    teacher=step_teacher,
    executor=tool_executor,
)
corrections = collector.collect(sample, student_actions)
~~~

The student-visible SFT input contains only query, previous actions, public
observation, and tool schema. The teacher receives the gold answer plus the
student call history/results; no separate verifier feedback is required by
default.

For online self-distillation, add the following fields to the row extra_info:

~~~python
{
    "opd_mm_online_self_distill": True,
    "opd_mm_step_teacher_class": "my_package.my_module.MyStepTeacher",
    "opd_mm_step_teacher_kwargs": {},  # optional
}
~~~

AgentLoopWorker.generate_sequences will then collect step corrections after the
student ToolAgentLoop rollout and store them in the returned non-tensor batch
field opd_mm_step_corrections.

## SFT conversion

Existing OPD-MM JSONL files with input and target fields can be converted to
verl multiturn-SFT parquet for warm start or ablation. This is not the main
on-policy distillation path:

~~~python
from verl.experimental.opd_mm.sft import convert_opd_sft_jsonl

convert_opd_sft_jsonl("sft_data.jsonl", "opd_mm_sft.parquet", include_tools=True)
~~~

include_tools=True attaches the OpenAI tool schemas to each row for tokenizers
that support tool-aware chat templates.

## Reward manager

The initial OPD-MM reward bridge can be selected through the importlib reward
manager path when distillation.distillation_loss.use_task_rewards=True. For
pure OPD, set use_task_rewards=False and rely on teacher logprobs:

~~~bash
reward.reward_manager.source=importlib \
reward.reward_manager.module.path=verl.experimental.opd_mm.reward_manager \
reward.reward_manager.name=OPDMMRewardManager
~~~

It reads correctness, support_recall, evidence_count, or evidence metadata from
the non-tensor batch / extra_info and places the scalar reward on the last
response token.

## Current scope

This migration provides the verl OPD environment adapters first. It does not
copy the standalone OmniMem argparse/accelerate training loops; training should
use verl's on-policy distillation trainer, agent-loop rollout, distillation
teacher server, and optional reward-manager infrastructure.
