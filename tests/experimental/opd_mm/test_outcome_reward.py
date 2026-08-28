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


def _request_kind(kwargs: dict[str, Any]) -> str:
    prompt = json.dumps(kwargs["messages"], ensure_ascii=False)
    if "public evidence alone" in prompt:
        return "evidence"
    if "sole correctness reference" in prompt:
        return "judge"
    return "answer"


def test_outcome_reward_generates_answer_before_gold_aware_judge(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        calls.append(kwargs)
        if _request_kind(kwargs) == "answer":
            return "The bicycle was red."
        if _request_kind(kwargs) == "evidence":
            return '{"answerable":true,"reason":"The color is explicit."}'
        return 'wrapper {"correct":true,"reason":"The answer matches."}'

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

    assert result["score"] == 1.2
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/evidence_answerable"] == 1.0
    assert result["opd_mm/outcome_evaluated"] == 1.0
    assert len(calls) == 3
    answer_prompt = json.dumps(next(call for call in calls if _request_kind(call) == "answer")["messages"], ensure_ascii=False)
    judge_prompt = json.dumps(next(call for call in calls if _request_kind(call) == "judge")["messages"], ensure_ascii=False)
    assert "It was red." not in answer_prompt
    assert "bright red" in answer_prompt
    assert "include every supported member" in answer_prompt
    assert "Return an image_id only when the question asks" in answer_prompt
    assert "ordinary factual questions" in answer_prompt
    assert "Not mentioned and No are distinct" in answer_prompt
    assert "No requires an explicit negative statement" in answer_prompt
    assert "a nearby action is not a cause" in answer_prompt
    assert "person, conversation, dialogue, or record mentioned" in answer_prompt
    assert "explicit incompatible alternative" in answer_prompt
    assert "It was red." in judge_prompt
    assert "The bicycle was red." in judge_prompt
    dumped = list(tmp_path.glob("outcome_reward_*.jsonl"))
    assert len(dumped) == 1
    row = json.loads(dumped[0].read_text(encoding="utf-8"))
    assert row["correct"] is True
    assert row["candidate_answer"] == "The bicycle was red."


def test_structured_conflict_relations_map_to_benchmark_verdicts() -> None:
    query = "Does this claim conflict with the dialogue memory?"
    assert outcome_reward._normalize_answer_output(query, '{"comparison":"INCOMPATIBLE_VALUE"}') == "Yes"
    assert outcome_reward._normalize_answer_output(query, '{"comparison":"NONE"}') == "No"
    assert (
        outcome_reward._normalize_answer_output(query, '{"comparison":"MISSING_COMPARISON"}')
        == "INSUFFICIENT_EVIDENCE"
    )

    claim_query = "Would it be accurate to say Evelyn always worked late?"
    assert outcome_reward._normalize_answer_output(
        claim_query,
        '{"comparison":"INCOMPATIBLE_ATTRIBUTE"}',
        "CD",
    ) == "No"
    assert outcome_reward._normalize_answer_output(
        claim_query,
        '{"comparison":"NONE"}',
        "CD",
    ) == "Yes"
    with pytest.raises(ValueError):
        outcome_reward._normalize_answer_output(query, "Yes")

    internal_query = "Does the conversation contain conflicting statements about the product?"
    assert outcome_reward._normalize_answer_output(
        internal_query,
        '{"comparison":"NONE","conflicting_pair_found":false}',
        "CD",
    ) == "No"
    assert outcome_reward._normalize_answer_output(
        internal_query,
        '{"comparison":"INCOMPATIBLE_IDENTITY","conflicting_pair_found":true}',
        "CD",
    ) == "Yes"


def test_cd_verdict_comparison_ignores_explanation_and_uses_normalized_label() -> None:
    assert outcome_reward._cd_verdict_matches_gold("No", "No.") is True
    assert outcome_reward._cd_verdict_matches_gold("Yes", "No.") is False
    assert outcome_reward._cd_verdict_matches_gold("INSUFFICIENT_EVIDENCE", "No.") is None


def test_cd_outcome_uses_structured_relation_without_a_second_model_judge(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        calls.append(kind)
        if kind == "answer":
            return (
                '{"question_proposition":"the image is Canton Tower",'
                '"memory_proposition":"the image is Oriental Pearl Tower",'
                '"comparison":"INCOMPATIBLE_IDENTITY","evidence_ids":["E1"]}'
            )
        if kind == "evidence":
            return '{"answerable":true,"reason":"explicit incompatible identity"}'
        raise AssertionError("structured CD verdict must not be reinterpreted by a free-text judge")

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="Yes.",
            extra_info={
                "point": "CD",
                "gold_answer": "Yes.",
                "opd_mm": _state(
                    query="Does the tower identity conflict with the dialogue memory?",
                    evidence=[{"evidence_id": "E1", "content": "The dialogue names the Canton Tower."}],
                ),
            },
        )
    )

    assert calls == ["answer", "evidence"]
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/outcome_evaluated"] == 1.0


@pytest.mark.parametrize("data_source", ["opd_mm_mmem_val", "opd_mm_memgallery_val"])
def test_outcome_reward_accepts_named_opd_mm_validation_sources(data_source, monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "The bicycle was red."
        if kind == "evidence":
            return '{"answerable":true,"reason":"explicit"}'
        return '{"correct":true,"reason":"matches"}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source=data_source,
            solution_str="",
            ground_truth="It was red.",
            extra_info={"opd_mm": _state(), "gold_answer": "It was red."},
        )
    )

    assert result["opd_mm/answer_correct"] == 1.0


def test_answer_judge_uses_evidence_only_to_validate_extra_details() -> None:
    messages = outcome_reward._judge_messages(
        query="Which conference was recommended?",
        gold_answer="AAAI.",
        evidence=[{"content": "Unrelated public evidence."}],
        candidate_answer="INSUFFICIENT_EVIDENCE",
    )
    prompt = json.dumps(messages, ensure_ascii=False)

    assert "sole correctness reference" in prompt
    assert "INSUFFICIENT_EVIDENCE is incorrect" in prompt
    assert "cannot replace a missing required answer" in prompt
    assert "AAAI." in prompt
    assert "Unrelated public evidence." in prompt


def test_answer_judge_distinguishes_unstated_events_from_explicit_negatives() -> None:
    messages = outcome_reward._judge_messages(
        query="What brand was the bicycle?",
        gold_answer="Not mentioned.",
        evidence=[],
        candidate_answer="INSUFFICIENT_EVIDENCE",
    )
    prompt = json.dumps(messages, ensure_ascii=False)

    assert "cannot be determined from the record is equivalent" in prompt
    assert "A bare No is equivalent only" in prompt
    assert "asks whether an event occurred" in prompt


def test_evidence_judge_requires_relevant_context_for_absence_answers() -> None:
    messages = outcome_reward._evidence_answerable_messages(
        query="What brand was the bicycle?",
        gold_answer="Not mentioned.",
        evidence=[{"content": "The bicycle was red."}],
    )
    prompt = json.dumps(messages, ensure_ascii=False)

    assert "question-visible subject and modality" in prompt
    assert "Unrelated or partial evidence is INSUFFICIENT" in prompt


def test_question_image_is_forwarded_to_answer_and_evidence_models(tmp_path) -> None:
    image = tmp_path / "question.jpg"
    image.write_bytes(b"jpeg-bytes")

    answer_messages = outcome_reward._answer_messages(
        "Which stored image matches this picture?",
        [{"evidence_id": "E1", "images": [{"image_id": "IMG1"}]}],
        question_image=str(image),
    )
    evidence_messages = outcome_reward._evidence_answerable_messages(
        "Which stored image matches this picture?",
        "IMG1",
        [{"evidence_id": "E1", "images": [{"image_id": "IMG1"}]}],
        question_image=str(image),
    )

    for messages in (answer_messages, evidence_messages):
        content = messages[-1]["content"]
        assert isinstance(content, list)
        assert content[-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_structured_evidence_support_classes_have_deterministic_polarity() -> None:
    assert outcome_reward._parse_answerable_with_recovery('{"support":"DIRECT"}')[0] is True
    assert outcome_reward._parse_answerable_with_recovery('{"support":"ABSENCE_WITH_COVERAGE"}')[0] is True
    assert outcome_reward._parse_answerable_with_recovery('{"support":"INSUFFICIENT"}')[0] is False
    assert outcome_reward._parse_answerable_with_recovery('{"support":"CONTRADICTED"}')[0] is False

    stalled = {
        "state": "stalled",
        "distinct_discovery_count": 2,
        "consecutive_no_gain_count": 2,
        "retrieval_methods_tried": ["bm25", "dense"],
        "modalities_searched": ["text"],
    }
    assert outcome_reward._parse_answerable_with_recovery(
        '{"support":"ABSENCE_WITH_COVERAGE"}',
        query="What brand was mentioned?",
        gold_answer="Not mentioned.",
        search_progress=stalled,
    )[0] is True
    assert outcome_reward._parse_answerable_with_recovery(
        '{"support":"ABSENCE_WITH_COVERAGE"}',
        query="What brand was mentioned?",
        gold_answer="Acme",
        search_progress=stalled,
    )[0] is False


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
    assert nonterminal["opd_mm/outcome_infrastructure_failure"] == 0.0
    assert empty["score"] == pytest.approx(-0.1)
    assert empty["opd_mm/outcome_evaluated"] == 0.0
    assert empty["opd_mm/outcome_infrastructure_failure"] == 0.0


def test_outcome_reward_evaluates_stalled_empty_absence_state(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        calls.append(kind)
        if kind == "answer":
            return "Not mentioned"
        if kind == "evidence":
            return '{"answerable":true,"reason":"Complementary searches stalled."}'
        return '{"correct":true,"reason":"Matches the absence answer."}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="Not mentioned.",
            extra_info={
                "gold_answer": "Not mentioned.",
                "opd_mm": _state(
                    query="What dessert was mentioned?",
                    evidence=[],
                    terminated=True,
                    policy_stopped=True,
                    termination_reason="policy_stop",
                    search_progress={
                        "state": "stalled",
                        "distinct_discovery_count": 2,
                        "consecutive_no_gain_count": 2,
                        "retrieval_methods_tried": ["bm25", "dense"],
                        "rewritten_query_count": 0,
                        "metadata_fields_tried": [],
                        "neighbor_windows_tried": [],
                        "modalities_searched": ["text"],
                    },
                ),
            },
        )
    )

    assert sorted(calls) == ["answer", "evidence", "judge"]
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/evidence_answerable"] == 1.0
    assert result["opd_mm/empty_evidence"] == 1.0
    assert result["opd_mm/search_stalled_rate"] == 1.0


def test_outcome_reward_applies_only_bounded_trajectory_penalties(monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "red"
        if kind == "evidence":
            return '{"answerable":true,"reason":"supported"}'
        return '{"correct":true,"reason":"supported"}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    repeated_trace = [
        {"tool": "SEARCH_METADATA", "field": "modality", "op": "eq", "value": "text"},
        {"tool": "SEARCH_METADATA", "field": "modality", "op": "eq", "value": "text"},
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

    assert result["score"] == pytest.approx(1.08)
    assert result["opd_mm/repeated_actions"] == 1.0
    assert result["opd_mm/max_actions_reached"] == 1.0


def test_outcome_judge_requires_boolean_correct() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        outcome_reward._parse_correct('{"correct": "TRUE", "reason": "invalid type"}')


def test_outcome_judge_recovers_unambiguous_boolean_from_truncated_json(monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "electric bass"
        if kind == "evidence":
            return '{"answerable":true,"reason":"supported"}'
        return '{"correct":true,"reason":"supported by E5"'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="electric bass",
            extra_info={"opd_mm": _state(), "gold_answer": "electric bass"},
        )
    )

    assert result["score"] == 1.2
    assert result["opd_mm/answer_correct"] == 1.0
    assert result["opd_mm/outcome_evaluated"] == 1.0
    assert result["opd_mm/judge_parse_recovered"] == 1.0
    assert result["opd_mm/judge_parse_failed"] == 0.0


def test_outcome_judge_invalid_output_is_conservative_not_fatal(monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "red"
        if kind == "evidence":
            return '{"answerable":false,"reason":"missing"}'
        return "I cannot produce JSON"

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
    assert result["opd_mm/outcome_infrastructure_failure"] == 1.0
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
    assert result["opd_mm/outcome_infrastructure_failure"] == 1.0
    assert result["opd_mm/answer_request_failed"] == 1.0


def test_empty_answer_response_is_an_infrastructure_failure(monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        if _request_kind(kwargs) == "answer":
            return "   "
        return '{"answerable":true,"reason":"supported"}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(), "gold_answer": "red"},
        )
    )

    assert result["opd_mm/outcome_evaluated"] == 0.0
    assert result["opd_mm/outcome_infrastructure_failure"] == 1.0
    assert result["opd_mm/answer_request_failed"] == 1.0


def test_outcome_dump_failure_does_not_fail_reward(tmp_path, monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "red"
        if kind == "evidence":
            return '{"answerable":true,"reason":"supported"}'
        return '{"correct":true,"reason":"supported"}'

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

    assert result["score"] == 1.2
    assert result["opd_mm/answer_correct"] == 1.0


def test_evidence_answerable_and_efficiency_reward_are_conditional(monkeypatch) -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        kind = _request_kind(kwargs)
        if kind == "answer":
            return "wrong answer"
        if kind == "evidence":
            return '{"answerable":true,"reason":"gold can be derived"}'
        return '{"correct":false,"reason":"does not match"}'

    monkeypatch.setattr(outcome_reward, "_chat_completion", fake_chat_completion)
    evidence = [{"content": f"memory {index}"} for index in range(20)]
    trace = [{"tool": "RETRIEVE", "method": "bm25", "top_k": 5}] * 4 + [{"tool": "STOP"}]
    result = asyncio.run(
        outcome_reward.compute_outcome_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state(evidence=evidence, trace=trace), "gold_answer": "red"},
            repeat_penalty=0.0,
        )
    )

    assert result["opd_mm/answer_correct"] == 0.0
    assert result["opd_mm/evidence_answerable"] == 1.0
    assert result["opd_mm/action_over_budget"] == 2.0
    assert result["opd_mm/evidence_over_budget"] == 4.0
    assert result["opd_mm/efficiency_penalty"] == pytest.approx(0.04)
    assert result["score"] == pytest.approx(0.16)


def test_opsd_reward_skips_outcome_calls_for_training(monkeypatch) -> None:
    async def unexpected_outcome(**kwargs: Any) -> dict[str, float]:
        del kwargs
        raise AssertionError("training rows must not invoke answer or judge")

    monkeypatch.setattr(outcome_reward, "compute_outcome_score", unexpected_outcome)
    result = asyncio.run(
        outcome_reward.compute_opsd_validation_score(
            data_source="opd_mm",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state()},
        )
    )

    assert result["score"] == 0.0
    assert result["opd_mm/outcome_evaluated"] == 0.0


def test_opsd_reward_routes_validation_to_answer_correctness(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_outcome(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"score": 1.0, "opd_mm/answer_correct": 1.0}

    monkeypatch.setattr(outcome_reward, "compute_outcome_score", fake_outcome)
    result = asyncio.run(
        outcome_reward.compute_opsd_validation_score(
            data_source="opd_mm_eval",
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state()},
        )
    )

    assert result["opd_mm/answer_correct"] == 1.0
    assert calls[0]["data_source"] == "opd_mm"


@pytest.mark.parametrize("data_source", ["opd_mm_mmem_val", "opd_mm_memgallery_val"])
def test_opsd_reward_routes_named_validation_sources(data_source, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_outcome(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"score": 1.0, "opd_mm/answer_correct": 1.0}

    monkeypatch.setattr(outcome_reward, "compute_outcome_score", fake_outcome)
    result = asyncio.run(
        outcome_reward.compute_opsd_validation_score(
            data_source=data_source,
            solution_str="",
            ground_truth="red",
            extra_info={"opd_mm": _state()},
        )
    )

    assert result["opd_mm/answer_correct"] == 1.0
    assert calls[0]["data_source"] == data_source
