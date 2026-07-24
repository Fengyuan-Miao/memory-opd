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

"""Terminal answer-correctness reward for OPD-MM GRPO.

The retrieval policy never sees the gold answer.  Once a trajectory terminates,
a fixed answer model consumes the public evidence and a separate judge compares
that generated answer with the private gold answer.  Both calls use an
OpenAI-compatible endpoint so the reward model can run outside the actor's Ray
resource pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_OUTCOME_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_OUTCOME_MODEL = "opd-mm-outcome"
logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _plain(value)
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if isinstance(value, list | tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else []
    return []


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(value or "").strip()


def _endpoint(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    candidates = [stripped]
    if "```" in stripped:
        for part in stripped.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].lstrip()
            if part:
                candidates.append(part)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError(f"judge did not return a JSON object: {stripped[:240]}")


def _parse_correct(text: str) -> tuple[bool, str]:
    correct, reason, _ = _parse_correct_with_recovery(text)
    return correct, reason


def _parse_correct_with_recovery(text: str) -> tuple[bool, str, bool]:
    """Parse the judge verdict, recovering only an unambiguous boolean.

    The judge sometimes truncates the final closing brace while still emitting
    a complete ``correct`` field. First try strict JSON, then accept a single,
    unambiguous JSON-style boolean literal. Never coerce quoted strings such as
    ``"TRUE"`` because that can hide prompt-following failures.
    """
    try:
        value = _json_object(text)
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("judge JSON field 'correct' must be a boolean")
        return correct, str(value.get("reason") or ""), False
    except ValueError as strict_error:
        matches = re.findall(r'["\']correct["\']\s*:\s*(true|false)\b', str(text or ""), flags=re.IGNORECASE)
        verdicts = {match.lower() == "true" for match in matches}
        if len(verdicts) != 1:
            raise strict_error
        return verdicts.pop(), "", True


def _answer_messages(query: str, evidence: list[Any]) -> list[dict[str, str]]:
    evidence_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    return [
        {
            "role": "system",
            "content": (
                "Answer the memory question using only the supplied public evidence. "
                "Do not use outside knowledge or the hidden memory store. If the evidence cannot support an answer, "
                "return exactly INSUFFICIENT_EVIDENCE. Return only the final answer."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nPublic evidence:\n{evidence_json}",
        },
    ]


def _judge_messages(
    query: str,
    gold_answer: str,
    evidence: list[Any],
    candidate_answer: str,
) -> list[dict[str, str]]:
    # Evidence quality is evaluated separately. Including evidence here can
    # make the judge reward a reasonable refusal after a failed retrieval,
    # even though that refusal does not answer the question or match gold.
    del evidence
    return [
        {
            "role": "system",
            "content": (
                "Judge answer correctness for a memory-QA benchmark. Use the gold answer as the sole correctness "
                "reference: mark correct only if the candidate directly answers the question and is semantically "
                "equivalent to the gold answer. Do not evaluate retrieval or evidence sufficiency. A refusal, unknown, "
                "or INSUFFICIENT_EVIDENCE is incorrect when the gold answer provides a substantive answer; it is "
                "correct only when the gold answer itself explicitly means unknown or not mentioned. "
                "Return only JSON: {\"correct\":true|false,\"reason\":\"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\nGold answer:\n{gold_answer}\n\n"
                f"Candidate answer:\n{candidate_answer}"
            ),
        },
    ]


async def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    timeout: float,
    max_tokens: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client_timeout = aiohttp.ClientTimeout(total=float(timeout))
    last_error: Exception | None = None

    for attempt in range(max(1, int(retries))):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(_endpoint(base_url), json=payload, headers=headers) as response:
                    body = await response.text()
                    if response.status >= 400:
                        error = RuntimeError(f"outcome service HTTP {response.status}: {body[:400]}")
                        if response.status < 500:
                            raise error
                        last_error = error
                    else:
                        parsed = json.loads(body)
                        choices = parsed.get("choices") if isinstance(parsed, dict) else None
                        if not choices or not isinstance(choices[0], dict):
                            raise RuntimeError(f"outcome service returned no choices: {body[:400]}")
                        message = choices[0].get("message") or {}
                        content = _content_text(message.get("content"))
                        if not content:
                            raise RuntimeError("outcome service returned empty content")
                        return content
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, RuntimeError) and "HTTP 4" in str(exc):
                break
        if attempt + 1 < max(1, int(retries)):
            await asyncio.sleep(min(2**attempt, 4))

    raise RuntimeError(f"outcome model request failed after {max(1, int(retries))} attempts: {last_error}")


def _repeat_count(trace: list[Any]) -> int:
    seen: set[str] = set()
    repeats = 0
    for item in trace:
        if not isinstance(item, dict):
            continue
        signature = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        if signature in seen:
            repeats += 1
        else:
            seen.add(signature)
    return repeats


async def _dump_result(payload: dict[str, Any]) -> None:
    dump_dir = str(os.getenv("OPD_MM_OUTCOME_REWARD_DUMP_DIR") or "").strip()
    if not dump_dir:
        return
    path = Path(dump_dir) / f"outcome_reward_{os.getpid()}.jsonl"
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    try:
        await asyncio.to_thread(write)
    except OSError as exc:
        # Reward dumps are diagnostic only. A full or temporarily unavailable
        # filesystem must not discard the computed reward or stop training.
        logger.warning("Failed to write OPD-MM outcome reward dump %s: %s", path, exc)


async def compute_outcome_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    answer_base_url: str | None = None,
    answer_model: str | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    answer_max_tokens: int = 256,
    judge_max_tokens: int = 192,
    retries: int = 3,
    repeat_penalty: float = 0.02,
    max_action_penalty: float = 0.1,
    error_penalty: float = 0.1,
    non_stop_penalty: float = 0.1,
    empty_evidence_penalty: float = 0.1,
    **kwargs: Any,
) -> dict[str, float]:
    """Generate and judge a final answer from an OPD-MM terminal state."""
    del solution_str, kwargs
    if data_source != "opd_mm":
        raise NotImplementedError(f"Outcome reward is not implemented for {data_source=}")

    info = _as_dict(extra_info)
    state = _as_dict(info.get("opd_mm"))
    evidence = _as_list(state.get("evidence"))
    trace = _as_list(state.get("trace"))
    query = str(state.get("query") or "").strip()
    if not query:
        tools_kwargs = _as_dict(info.get("tools_kwargs"))
        query = str(_as_dict(tools_kwargs.get("opd_mm")).get("query") or "").strip()
    gold_answer = str(_plain(info.get("gold_answer", ground_truth)) or "").strip()

    stopped = bool(state.get("stopped"))
    trajectory_error = bool(state.get("error"))
    max_actions_reached = bool(state.get("max_actions_reached"))
    repeats = _repeat_count(trace)
    correct = False
    evaluated = False
    candidate_answer = ""
    judge_raw = ""
    judge_reason = ""
    outcome_error = ""
    answer_request_failed = False
    judge_request_failed = False
    judge_parse_recovered = False
    judge_parse_failed = False

    if stopped and evidence and query and gold_answer and not trajectory_error:
        outcome_url = answer_base_url or os.getenv("OPD_MM_OUTCOME_BASE_URL") or DEFAULT_OUTCOME_BASE_URL
        outcome_model = answer_model or os.getenv("OPD_MM_OUTCOME_MODEL") or DEFAULT_OUTCOME_MODEL
        judge_url = judge_base_url or os.getenv("OPD_MM_JUDGE_BASE_URL") or outcome_url
        judge_model_name = judge_model or os.getenv("OPD_MM_JUDGE_MODEL") or outcome_model
        outcome_api_key = api_key or os.getenv("OPD_MM_OUTCOME_API_KEY") or ""

        try:
            candidate_answer = await _chat_completion(
                base_url=outcome_url,
                model=outcome_model,
                messages=_answer_messages(query, evidence),
                api_key=outcome_api_key,
                timeout=float(timeout),
                max_tokens=int(answer_max_tokens),
                retries=int(retries),
            )
        except Exception as exc:
            answer_request_failed = True
            outcome_error = f"answer_request_failed:{type(exc).__name__}:{exc}"[:500]

        if candidate_answer:
            parse_error: Exception | None = None
            for judge_attempt in range(max(1, int(retries))):
                try:
                    judge_raw = await _chat_completion(
                        base_url=judge_url,
                        model=judge_model_name,
                        messages=_judge_messages(query, gold_answer, evidence, candidate_answer),
                        api_key=outcome_api_key,
                        timeout=float(timeout),
                        max_tokens=int(judge_max_tokens),
                        retries=int(retries),
                    )
                    correct, judge_reason, recovered = _parse_correct_with_recovery(judge_raw)
                    judge_parse_recovered = judge_parse_recovered or recovered
                    parse_error = None
                    evaluated = True
                    break
                except ValueError as exc:
                    parse_error = exc
                except Exception as exc:
                    judge_request_failed = True
                    parse_error = exc
                if judge_attempt + 1 < max(1, int(retries)):
                    await asyncio.sleep(min(2**judge_attempt, 4))
            if parse_error is not None:
                judge_parse_failed = not judge_request_failed
                outcome_error = f"judge_failed:{type(parse_error).__name__}:{parse_error}"[:500]

    score = float(correct)
    score -= float(repeat_penalty) * repeats
    score -= float(max_action_penalty) if max_actions_reached else 0.0
    score -= float(error_penalty) if trajectory_error else 0.0
    score -= float(non_stop_penalty) if not stopped else 0.0
    score -= float(empty_evidence_penalty) if stopped and not evidence else 0.0
    score = max(-1.0, min(1.0, score))

    await _dump_result(
        {
            "query": query,
            "gold_answer": gold_answer,
            "evidence": evidence,
            "trace": trace,
            "candidate_answer": candidate_answer,
            "judge_raw": judge_raw,
            "judge_reason": judge_reason,
            "outcome_error": outcome_error,
            "correct": correct,
            "score": score,
            "stopped": stopped,
            "max_actions_reached": max_actions_reached,
            "trajectory_error": state.get("error") or "",
            "repeat_count": repeats,
            "drop_calls": int(state.get("drop_calls") or 0),
            "dropped_evidence_count": int(state.get("dropped_evidence_count") or 0),
            "answer_request_failed": answer_request_failed,
            "judge_request_failed": judge_request_failed,
            "judge_parse_recovered": judge_parse_recovered,
            "judge_parse_failed": judge_parse_failed,
        }
    )

    return {
        "score": score,
        "opd_mm/answer_correct": float(correct),
        "opd_mm/outcome_evaluated": float(evaluated),
        "opd_mm/terminal_stopped": float(stopped),
        "opd_mm/evidence_count": float(len(evidence)),
        "opd_mm/empty_evidence": float(not evidence),
        "opd_mm/repeated_actions": float(repeats),
        "opd_mm/drop_calls": float(state.get("drop_calls") or 0),
        "opd_mm/dropped_evidence_count": float(state.get("dropped_evidence_count") or 0),
        "opd_mm/max_actions_reached": float(max_actions_reached),
        "opd_mm/trajectory_error": float(trajectory_error),
        "opd_mm/answer_request_failed": float(answer_request_failed),
        "opd_mm/judge_request_failed": float(judge_request_failed),
        "opd_mm/judge_parse_recovered": float(judge_parse_recovered),
        "opd_mm/judge_parse_failed": float(judge_parse_failed),
    }


__all__ = ["compute_outcome_score"]
