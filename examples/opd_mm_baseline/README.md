# OPD-MM baseline in verl

This example hosts the migrated OPD-MM hidden-memory retrieval environment.
It is intentionally split into two layers:

- verl.experimental.opd_mm: CPU-only environment, validator, retriever,
  executor, verl-native tools, SFT conversion helpers, and reward bridge.
- examples/opd_mm_baseline: wiring examples for verl configs and smoke runs.

The migration keeps the original OPD-MM isolation contract:

- the policy sees the user query and public tool schemas;
- the hidden memory store is available only to the OPD-MM tools/executor;
- tool observations omit memory IDs;
- retrieve always uses the original user query and rejects custom query fields
  through the validator.

## verl-native tool path

Use opd_mm_tool_config.yaml as the rollout multi-turn tool config:

~~~bash
actor_rollout_ref.rollout.multi_turn.enable=true \
actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/opd_mm_baseline/opd_mm_tool_config.yaml
~~~

Per sample, pass hidden memory state through the non-tensor field tools_kwargs,
for example:

~~~python
{
    "raw_prompt": [{"role": "user", "content": "Which pet image did I upload last?"}],
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

## SFT conversion

Existing OPD-MM JSONL files with input and target fields can be converted to
verl multiturn-SFT parquet:

~~~python
from verl.experimental.opd_mm.sft import convert_opd_sft_jsonl

convert_opd_sft_jsonl("sft_data.jsonl", "opd_mm_sft.parquet", include_tools=True)
~~~

include_tools=True attaches the OpenAI tool schemas to each row for tokenizers
that support tool-aware chat templates.

## Reward manager

The initial OPD-MM reward bridge can be selected through the importlib reward
manager path:

~~~bash
reward.reward_manager.source=importlib \
reward.reward_manager.module.path=verl.experimental.opd_mm.reward_manager \
reward.reward_manager.name=OPDMMRewardManager
~~~

It reads correctness, support_recall, evidence_count, or evidence metadata from
the non-tensor batch / extra_info and places the scalar reward on the last
response token.

## Current scope

This migration provides the verl environment adapters first. It does not copy
the standalone OmniMem argparse/accelerate training loops; training should use
verl SFT, rollout, agent-loop, and reward-manager infrastructure.
