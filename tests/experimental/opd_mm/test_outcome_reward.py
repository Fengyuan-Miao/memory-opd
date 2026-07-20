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

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from verl.experimental.opd_mm import outcome_reward


def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "query": "What color was the bicycle?",
        "evidence": [{"content": "The bicycle beside the door was bright red."}],
        "trace": [
            {"tool": "RETRIEVE", "method": "dense", "top_k": 5},
            {"tool": "STOP"},
        ],
        "stopped": True,
        "error": "",
        "max_actions_reached": False,
    }
    state.update(overrides)
    return state


def test_outcome_reward_generates_answer_before_gold_aware_judge(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            return "The bicycle was red."
        return 'wrapper {"correct": true, "reason": "The answer matches and is supported."}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    monkeypatch.setenv("OPD_MM_OUTCOME_REWARD_DUMP_DIR", str(tmp_path))
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="<tool_call>stop</tool_call>",
            ground_truth="It was red.",
            extra_info={"opd_mm": _state(), "gold_answer": "It was red."},
        )
    )

    assert result["score"] == 1.0
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/outcome_evaluated"] == 1.0
    assert len(calls) == 2
    answer_prompt = json.dumps(calls[0]["messages"], ensure_ascii=False)
    judge_prompt = json.dumps(calls[1]["messages"], ensure_ascii=False)
    assert "It was red." not in answer_prompt
    assert "bright red" in answer_prompt
    assert "It was red." in judge_prompt
    assert "The bicycle was red." in judge_prompt
    dumped = list(tmp_path.glob("outcome_reward_*.jsonl"))
    assert len(dumped) == 1
    row = json.loads(dumped[0].read_text(encoding="utf-8"))
    assert row["correct"] is True
    assert row["candidate_answer"] == "The bicycle was red."


def test_outcome_reward_does_not_call_models_for_nonterminal_or_empty_evidence(monkeypatch) -> None:
    async def unexpected_call(**kwargs: Any) -> str:
        del kwargs
        raise AssertionError("outcome model should not be called")

    monkeypatch.setattr(outcome_reward, "_chat_completion", unexpected_call)
    nonterminal = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(stopped=False)},
        )
    )
    empty = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(evidence=[])},
        )
    )

    assert nonterminal["score"] == pytest.approx(-0.1)
    assert nonterminal["opd_mm/outcome_evaluated"] == 0.0
    assert empty["score"] == pytest.approx(-0.1)
    assert empty["opd_mm/outcome_evaluated"] == 0.0


def test_outcome_reward_applies_only_bounded_trajectory_penalties(monkeypatch) -> None:
    replies = iter(["red", '{"correct": true, "reason": "supported"}'])

    async def fake_chat_completion(**kwargs: Any) -> str:
        del kwargs
        return next(replies)

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    repeated_trace = [
        {"tool": "FILTER", "field": "modality", "op": "eq", "value": "text"},
        {"tool": "FILTER", "field": "modality", "op": "eq", "value": "text"},
        {"tool": "STOP"},
    ]
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(trace=repeated_trace, max_actions_reached=True)},
        )
    )

    assert result["score"] == pytest.approx(0.88)
    assert result["opd_mm/repeated_actions"] == 1.0
    assert result["opd_mm/max_actions_reached"] == 1.0


def test_outcome_judge_requires_boolean_correct() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        outcome_reward._parse_correct('{"correct": "TRUE", "reason": "invalid type"}')


def test_outcome_judge_recovers_unambiguous_boolean_from_truncated_json(monkeypatch) -> None:
    replies = iter(
        [
            "electric bass",
            '{"correct":true,"reason":"supported by E5"',
        ]
    )

    async def fake_chat_completion(**kwargs: Any) -> str:
        del kwargs
        return next(replies)

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="electric bass",
            extra_info={"opd_mm": _state(), "gold_answer": "electric bass"},
        )
    )

    assert result["score"] == 1.0
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/outcome_evaluated"] == 1.0
    assert result["opd_mm/judge_parse_recovered"] == 1.0
    assert result["opd_mm/judge_parse_failed"] == 0.0


def test_outcome_judge_invalid_output_is_conservative_not_fatal(monkeypatch) -> None:
    calls = 0

    async def fake_chat_completion(**kwargs: Any) -> str:
        nonlocal calls
        del kwargs
        calls += 1
        return "red" if calls == 1 else "I cannot produce JSON"

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(), "gold_answer": "red"},
            retries=2,
        )
    )

    assert result["score"] == 0.0
    assert result["opd_mm/answer_correct"] == 0.0
    assert result["opd_mm/outcome_evaluated"] == 0.0
    assert result["opd_mm/judge_parse_failed"] == 1.0
    assert result["opd_mm/judge_request_failed"] == 0.0


def test_outcome_service_failure_is_conservative_not_fatal(monkeypatch) -> None:
    async def failed_chat_completion(**kwargs: Any) -> str:
        del kwargs
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(outcome_reward, "_chat_completion", failed_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(), "gold_answer": "red"},
        )
    )

    assert result["score"] == 0.0
    assert result["opd_mm/outcome_evaluated"] == 0.0
    assert result["opd_mm/answer_request_failed"] == 1.0


def test_outcome_dump_failure_does_not_fail_reward(tmp_path, monkeypatch) -> None:
    replies = iter(["red", '{"correct":true,"reason":"supported"}'])

    async def fake_chat_completion(**kwargs: Any) -> str:
        del kwargs
        return next(replies)

    def failed_open(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    monkeypatch.setattr(outcome_reward.Path, "open", failed_open)
    monkeypatch.setenv("OPD_MM_OUTCOME_REWARD_DUMP_DIR", str(tmp_path))

    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(), "gold_answer": "red"},
        )
    )

    assert result["score"] == 1.0
    assert result["opd_mm/answer_correct"] == 1.0
