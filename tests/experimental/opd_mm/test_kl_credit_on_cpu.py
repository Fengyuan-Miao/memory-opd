# Copyright 2025 Individual Contributor: Fengyuan Miao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.opd_mm.kl_credit import normalized_topk_union_kl, structured_action_disagreement
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_normalized_topk_union_kl_detects_distribution_disagreement() -> None:
    identical = normalized_topk_union_kl([1, 2], [-0.1, -2.0], [1, 2], [-0.1, -2.0])
    different = normalized_topk_union_kl([1, 2], [-0.1, -2.0], [3, 2], [-0.1, -2.0])

    assert identical == pytest.approx(0.0)
    assert different > 1.0


def test_structured_action_disagreement_covers_arguments_and_schema_errors() -> None:
    assert structured_action_disagreement(
        {"tool": "RETRIEVE"},
        {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
    ) == (False, "none")
    assert structured_action_disagreement(
        {"tool": "RETRIEVE", "method": "bm25", "top_k": 5},
        {"tool": "RETRIEVE", "method": "dense", "top_k": 5},
    ) == (True, "arguments")
    disagreement, reason = structured_action_disagreement(
        {"tool": "RETRIEVE", "method": "hm25", "top_k": 5},
        {"tool": "RETRIEVE", "method": "bm25", "top_k": 5},
    )
    assert disagreement is True
    assert reason.startswith("student_invalid:")


@pytest.mark.asyncio
async def test_state_kl_scores_teacher_and_student_on_the_student_action_prefix() -> None:
    class StudentClient:
        def __init__(self) -> None:
            self.sequence = None

        async def generate(self, **kwargs):
            self.sequence = kwargs["prompt_ids"]
            return SimpleNamespace(
                extra_fields={
                    "prompt_ids": [[0, 1], [7, 8], [9, 10], [0, 0]],
                    "prompt_logprobs": [[0.0, 0.0], [-0.1, -2.0], [-0.1, -2.0], [0.0, 0.0]],
                }
            )

    class TeacherClient:
        def __init__(self) -> None:
            self.sequence = None

        async def compute_teacher_logprobs_single(self, **kwargs):
            self.sequence = kwargs["sequence_ids"]
            return (
                torch.tensor([[0, 1], [0, 1], [7, 8], [10, 9], [0, 0]]),
                torch.tensor([[0.0, 0.0], [0.0, 0.0], [-0.1, -2.0], [-0.1, -2.0], [0.0, 0.0]]),
            )

    student = StudentClient()
    teacher = TeacherClient()
    worker = SimpleNamespace(
        llm_client=student,
        teacher_server_manager=teacher,
        distillation_config=SimpleNamespace(
            distillation_loss=SimpleNamespace(topk=2, log_prob_min_clamp=-10.0)
        ),
    )
    credit = await AgentLoopWorker._compute_opd_mm_state_kl_credit(
        worker,
        request={
            "student_prompt_ids": [1, 2],
            "student_response_ids": [3, 4],
            "student_tool_call_mask": [1, 1],
        },
        teacher_prompt_ids=[5, 6, 7],
        multi_modal_data={},
        mm_processor_kwargs={},
        routing_key=None,
    )

    assert student.sequence == [1, 2, 3, 4]
    assert teacher.sequence == [5, 6, 7, 3, 4]
    assert credit is not None
    assert credit["teacher_ids"] == [[7, 8], [10, 9]]
    assert credit["action_kl"] > 0.0


def _state(step: int, base: int) -> dict:
    return {
        "step_index": step,
        "prompt_ids": [base, base + 1],
        "response_ids": [base + 2, base + 3],
        "response_logprobs": [-0.1, -0.2],
        "tool_call_mask": [1, 0],
        "student_next_action": {"tool": "FILTER", "field": "modality", "op": "eq", "value": "text", "scope": "current_pool"},
    }


def _correction(step: int, score: float) -> dict:
    return {
        "step_index": step,
        "teacher_actions": [{"tool": "STOP"}],
        "kl_credit": {
            "structured_disagreement": True,
            "disagreement_type": "tool",
            "action_kl": score,
            "tool_call_mask": [1, 0],
            "teacher_ids": [[10, 11], [12, 13]],
            "teacher_logprobs": [[-0.1, -2.0], [-0.2, -1.8]],
        },
    }


def test_kl_credit_batch_routes_success_groups_to_grpo_and_all_fail_groups_to_distillation() -> None:
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            opd_mm_kl_credit={"enabled": True, "top_actions": 2, "success_key": "opd_mm/answer_correct"}
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=6, response_length=4, n=2),
            actor=SimpleNamespace(ppo_mini_batch_size=1),
        ),
    )
    trainer = SimpleNamespace(
        config=config,
        tokenizer=SimpleNamespace(pad_token_id=0),
        actor_rollout_wg=object(),
        _get_dp_size=lambda worker_group, role: 1,
    )
    states = np.empty(4, dtype=object)
    corrections = np.empty(4, dtype=object)
    for index in range(4):
        states[index] = [_state(0, 20 + index * 10)]
        corrections[index] = [_correction(0, 1.0 + index)]
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor([[1, 1, 0, 0]] * 4),
            "advantages": torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [-1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "token_level_scores": torch.zeros(4, 4),
        },
        non_tensors={
            "uid": np.array(["success-group", "success-group", "all-fail-group", "all-fail-group"], dtype=object),
            "opd_mm/answer_correct": np.array([1.0, 0.0, 0.0, 0.0]),
            "opd_mm_policy_states": states,
            "opd_mm_step_corrections": corrections,
        },
    )

    result = RayPPOTrainer._build_opd_mm_kl_credit_batch(trainer, batch)

    assert result is not None
    credit_batch, metrics = result
    assert credit_batch.non_tensor_batch["opd_mm_kl_credit_mode"].tolist() == [
        "grpo",
        "grpo",
        "distill",
        "distill",
    ]
    assert credit_batch.batch["response_mask"][:, 0].tolist() == [1, 1, 0, 0]
    assert credit_batch.batch["distillation_mask"][:, 0].tolist() == [0, 0, 1, 1]
    assert credit_batch.batch["advantages"][:, 0].tolist() == [1.0, -1.0, 0.0, 0.0]
    assert credit_batch.batch["teacher_ids"][0, 5].tolist() == [10, 11]
    assert metrics["opd_mm_grpo_states"] == 2.0
    assert metrics["opd_mm_distill_states"] == 2.0
    assert metrics["opd_mm_all_fail_groups"] == 1.0


def test_kl_credit_batch_keeps_only_the_two_highest_kl_actions_per_trajectory() -> None:
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            opd_mm_kl_credit={"enabled": True, "top_actions": 2, "success_key": "opd_mm/answer_correct"}
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=6, response_length=4, n=1),
            actor=SimpleNamespace(ppo_mini_batch_size=1),
        ),
    )
    trainer = SimpleNamespace(
        config=config,
        tokenizer=SimpleNamespace(pad_token_id=0),
        actor_rollout_wg=object(),
        _get_dp_size=lambda worker_group, role: 1,
    )
    states = np.empty(1, dtype=object)
    corrections = np.empty(1, dtype=object)
    states[0] = [_state(0, 20), _state(1, 30), _state(2, 40)]
    corrections[0] = [_correction(0, 0.2), _correction(1, 2.0), _correction(2, 0.8)]
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor([[1, 1, 0, 0]]),
            "advantages": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            "token_level_scores": torch.zeros(1, 4),
        },
        non_tensors={
            "uid": np.array(["success-group"], dtype=object),
            "opd_mm/answer_correct": np.array([1.0]),
            "opd_mm_policy_states": states,
            "opd_mm_step_corrections": corrections,
        },
    )

    result = RayPPOTrainer._build_opd_mm_kl_credit_batch(trainer, batch)

    assert result is not None
    credit_batch, metrics = result
    selected_steps = [item["step_index"] for item in credit_batch.non_tensor_batch["opd_mm_step_correction"]]
    assert selected_steps == [1, 2]
    assert metrics["opd_mm_kl_selected_actions"] == 2.0
    assert metrics["opd_mm_selected_action_kl"] == pytest.approx(1.4)
