#!/usr/bin/env python3
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

"""Evaluate OPD-MM rollouts with an external LLM judge.

The script intentionally mirrors the verl-native ToolAgentLoop semantics in a
lightweight offline runner:

1. load a HF/vLLM student checkpoint;
2. run multi-step OPD-MM tool calls over hidden Mem-Gallery memory records;
3. ask an external answer model to answer from the collected evidence;
4. ask a separate judge model whether the answer matches the gold answer.

It is meant for quick checkpoint evaluation rather than training.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.opd_mm.dataset import OPD_MM_SYSTEM_PROMPT, opd_messages_for_state
from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.models import ToolAction
from verl.experimental.opd_mm.retrieval import TurnAwareHybridRetriever
from verl.experimental.opd_mm.schema import TrajectoryValidator
from verl.experimental.opd_mm.tools import OPDToolSession, hidden_store_from_records, openai_tool_schemas


DEFAULT_STUDENT = (
    "checkpoints/verl_distill_opd_mm/"
    "opd_mm_qwen35_4b_frozen_teacher_mem_gallery_cap2_schemafix/"
    "global_step_105/actor_merged_hf_vllm"
)
DEFAULT_JUDGE = "/home/guojr/data/pretrained_models/Qwen/Qwen3.5-9B"
DEFAULT_QAS = "dataset/mem_gallery/opd_mm_store/qas.parquet"
DEFAULT_TRAIN_IDS = (
    "dataset/mem_gallery/opd_mm_store/subsets/balanced_train_cap2/"
    "train_sample_ids.txt"
)
DEFAULT_TRAIN_RLHF = (
    "dataset/mem_gallery/opd_mm_store/subsets/balanced_train_cap2/"
    "train.parquet"
)
DEFAULT_OUTPUT = "outputs/opd_mm_eval/llm_judge_eval.jsonl"
DEFAULT_ANSWER_BASE_URL = "http://192.168.1.113:31208"
DEFAULT_ANSWER_MODEL = "Qwen3-VL-4B-Instruct"


def _read_train_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, tuple) else []


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _load_records_by_scenario(train_rlhf_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    df = pd.read_parquet(train_rlhf_path)
    records_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in df.to_dict("records"):
        extra = row.get("extra_info") or {}
        scenario = str(extra.get("scenario") or "")
        opd_kwargs = ((extra.get("tools_kwargs") or {}).get("opd_mm") or {})
        records = _as_list(opd_kwargs.get("records"))
        if scenario and records and scenario not in records_by_scenario:
            records_by_scenario[scenario] = records
    return records_by_scenario


def _stratified_sample(
    qas: list[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample held-out QAs across scenario and QA category cells."""
    if max_samples <= 0 or max_samples >= len(qas):
        return sorted(qas, key=lambda x: (str(x.get("scenario")), str(x.get("point")), str(x.get("sample_id"))))

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for qa in qas:
        groups[(str(qa.get("scenario") or ""), str(qa.get("point") or ""))].append(qa)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    rng.shuffle(keys)
    while len(selected) < max_samples and keys:
        next_keys = []
        for key in keys:
            rows = groups[key]
            if not rows:
                continue
            rows.sort(key=lambda x: str(x.get("sample_id")))
            pick_index = rng.randrange(len(rows))
            selected.append(rows.pop(pick_index))
            if rows:
                next_keys.append(key)
            if len(selected) >= max_samples:
                break
        keys = next_keys
    return sorted(selected, key=lambda x: (str(x.get("scenario")), str(x.get("point")), str(x.get("sample_id"))))


def load_eval_qas(args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.read_parquet(args.qas_path)
    rows = df.to_dict("records")
    train_ids = _read_train_ids(args.train_sample_ids)
    heldout = [row for row in rows if str(row.get("sample_id")) not in train_ids]
    return _stratified_sample(heldout, max_samples=args.max_samples, seed=args.seed)


def _messages_for_query(query: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": OPD_MM_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _openai_endpoint(base_url: str, suffix: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith(suffix):
        return base
    if base.endswith("/v1"):
        return f"{base}{suffix.removeprefix('/v1')}"
    return f"{base}{suffix}"


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


def _read_openai_json(req: urllib_request.Request, timeout: float) -> dict[str, Any]:
    try:
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
        with opener.open(req, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"request to {req.full_url} failed: {exc}") from exc
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(f"OpenAI-compatible error from {req.full_url}: {parsed['error']}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected JSON response from {req.full_url}: {type(parsed).__name__}")
    return parsed


def _post_openai_json(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        _openai_endpoint(base_url, "/v1/chat/completions"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _read_openai_json(req, timeout)


def _discover_openai_model(base_url: str, timeout: float) -> str:
    req = urllib_request.Request(_openai_endpoint(base_url, "/v1/models"), method="GET")
    response = _read_openai_json(req, min(float(timeout), 10.0))
    data = response.get("data") if isinstance(response, dict) else None
    if isinstance(data, list) and data:
        model_id = data[0].get("id") if isinstance(data[0], dict) else None
        if model_id:
            return str(model_id)
    return ""


async def _parse_tool_calls(parser: Any, tokenizer: Any, text: str, tool_schemas: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    schema_objs = [
        # ToolParser expects pydantic schemas in the same shape used by verl.
        __import__("verl.tools.schemas", fromlist=["OpenAIFunctionToolSchema"]).OpenAIFunctionToolSchema.model_validate(
            schema
        )
        for schema in tool_schemas
    ]
    return await parser.extract_tool_calls(ids, schema_objs)


def _format_prompt(tokenizer: Any, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _make_session(qa: dict[str, Any], records: list[dict[str, Any]]) -> OPDToolSession:
    return OPDToolSession(
        executor=ToolExecutor(
            retriever=TurnAwareHybridRetriever(),
            validator=TrajectoryValidator(max_actions=8, max_top_k=50, allow_inspect_raw=True),
            max_raw_inspections=3,
        ),
        memory_store=hidden_store_from_records(records),
        query=str(qa.get("question") or ""),
        question_image=_optional_path(qa.get("question_image")),
    )


def rollout_student(args: argparse.Namespace, qas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_scenario = _load_records_by_scenario(args.train_rlhf_path)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    parser = ToolParser.get_tool_parser(args.tool_format, tokenizer)
    tool_schemas = openai_tool_schemas(include_inspect_raw=True)

    llm = LLM(
        model=args.student_model,
        trust_remote_code=True,
        tensor_parallel_size=args.student_tp,
        gpu_memory_utilization=args.student_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>"],
        include_stop_str_in_output=True,
    )

    outputs: list[dict[str, Any]] = []
    for qa in tqdm(qas, desc="student rollout"):
        sample_id = str(qa.get("sample_id"))
        scenario = str(qa.get("scenario") or "")
        records = records_by_scenario.get(scenario)
        if not records:
            outputs.append(
                {
                    "sample_id": sample_id,
                    "scenario": qa.get("scenario"),
                    "point": qa.get("point"),
                    "question": qa.get("question"),
                    "gold_answer": qa.get("gold_answer") or qa.get("answer"),
                    "error": f"missing scenario records for {scenario}",
                    "student_answer": "",
                    "trace": [],
                    "evidence": [],
                }
            )
            continue

        base_messages = _messages_for_query(str(qa.get("question") or ""))
        messages = list(base_messages)
        session = _make_session(qa, records)
        final_answer = ""
        raw_generations = []
        generation_error = ""
        prompt_token_lengths = []

        for _turn in range(args.max_turns):
            prompt = _format_prompt(tokenizer, messages, tool_schemas)
            prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_token_lengths.append(prompt_token_len)
            if prompt_token_len >= args.max_model_len:
                generation_error = (
                    f"prompt_too_long: prompt_tokens={prompt_token_len} "
                    f">= max_model_len={args.max_model_len}"
                )
                final_answer = raw_generations[-1].strip() if raw_generations else ""
                break
            result = llm.generate([prompt], sampling, use_tqdm=False)[0]
            text = result.outputs[0].text
            if text.endswith("<|im_end|>"):
                text = text[: -len("<|im_end|>")]
            raw_generations.append(text)

            _content, calls = asyncio.run(_parse_tool_calls(parser, tokenizer, text, tool_schemas))
            if not calls:
                final_answer = text.strip()
                break

            # ToolAgentLoop only executes max_parallel_calls=1 in the training config.
            calls = calls[:1]
            call = calls[0]
            try:
                action = ToolAction(call.name.upper(), json.loads(call.arguments or "{}"))
                observation = session.execute(action)
            except Exception as exc:  # keep the trajectory inspectable
                observation = {
                    "tool": getattr(call, "name", ""),
                    "pool_count": len(session.pool),
                    "evidence_count": len(session.evidence),
                    "pool_preview": [],
                    "new_evidence_count": 0,
                    "evidence_preview": [],
                    "stopped": session.stopped,
                    "error": str(exc),
                }
            messages = opd_messages_for_state(
                base_messages,
                [item.to_dict() for item in session.trace],
                observation,
            )
            if session.stopped:
                # STOP is the terminal action for the retrieval policy. The
                # retrieval model is not asked to produce the final QA answer;
                # an external answer model consumes the evidence afterward.
                final_answer = ""
                break
        else:
            final_answer = raw_generations[-1].strip() if raw_generations else ""

        state = session.public_state()
        outputs.append(
            {
                "sample_id": sample_id,
                "scenario": qa.get("scenario"),
                "point": qa.get("point"),
                "question": qa.get("question"),
                "gold_answer": qa.get("gold_answer") or qa.get("answer"),
                "student_answer": final_answer,
                "retrieval_model_answer": final_answer,
                "answer_contains_gold": float(
                    bool(_normalize_text(qa.get("gold_answer") or qa.get("answer")))
                    and _normalize_text(qa.get("gold_answer") or qa.get("answer")) in _normalize_text(final_answer)
                ),
                "trace": state.get("trace") or [],
                "evidence": state.get("evidence") or [],
                "evidence_count": state.get("evidence_count", 0),
                "pool_count": state.get("pool_count", 0),
                "num_turns": len(state.get("trace") or []),
                "max_prompt_tokens": max(prompt_token_lengths) if prompt_token_lengths else 0,
                "prompt_token_lengths": prompt_token_lengths,
                "raw_generations": raw_generations,
                "error": generation_error or state.get("error") or "",
            }
        )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return outputs


def _answer_prompt(row: dict[str, Any], max_evidence: int) -> list[dict[str, str]]:
    evidence = row.get("evidence") or []
    evidence_text = json.dumps(evidence[: max(0, int(max_evidence))], ensure_ascii=False, indent=2)
    system = (
        "You are the answer model for a memory QA benchmark. "
        "Answer the user's question using only the retrieved evidence. "
        "Do not mention tool calls. Do not use any gold answer. "
        "If the evidence is insufficient, answer with the best supported response and briefly state uncertainty."
    )
    user = f"""Question:
{row.get('question')}

Retrieved evidence:
{evidence_text}

Give the final answer only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_answer_model(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    model = str(args.answer_model or "").strip()
    if not model:
        try:
            model = _discover_openai_model(args.answer_base_url, args.answer_timeout) or "default"
        except Exception:
            model = "default"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(args.answer_temperature),
        "max_tokens": int(args.answer_max_new_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = _post_openai_json(args.answer_base_url, payload, args.answer_timeout)
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise RuntimeError(f"empty response from answer model service: {args.answer_base_url}")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    answer = _content_text(message.get("content"))
    if not answer:
        raise RuntimeError(f"empty answer content from answer model service: {args.answer_base_url}")
    return answer


def run_answer_model(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answered: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="answer model"):
        retrieval_model_answer = row.get("retrieval_model_answer", row.get("student_answer", ""))
        try:
            answer = _call_answer_model(args, _answer_prompt(row, args.answer_max_evidence))
            answer_error = ""
        except Exception as exc:
            answer = ""
            answer_error = f"{type(exc).__name__}: {exc}"
        gold_answer = row.get("gold_answer")
        answered.append(
            {
                **row,
                "retrieval_model_answer": retrieval_model_answer,
                "student_answer": answer,
                "answer_model_answer": answer,
                "answer_model_error": answer_error,
                "answer_contains_gold": float(bool(_normalize_text(gold_answer)) and _normalize_text(gold_answer) in _normalize_text(answer)),
            }
        )
    return answered


def _judge_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    evidence = row.get("evidence") or []
    evidence_text = json.dumps(evidence[:8], ensure_ascii=False, indent=2)
    system = (
        "You are a strict but fair evaluator for a memory QA benchmark. "
        "Judge whether the student's final answer correctly answers the question. "
        "Use the gold answer as the primary reference; evidence is provided only for context. "
        "Return only valid JSON with keys: correct (boolean), score (0 or 1), reason (short string)."
    )
    user = f"""Question:
{row.get('question')}

Gold answer:
{row.get('gold_answer')}

Student answer:
{row.get('student_answer')}

Retrieved evidence:
{evidence_text}

Is the student answer semantically correct?"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _evidence_answerable_prompt(row: dict[str, Any], max_evidence: int) -> list[dict[str, str]]:
    evidence = row.get("evidence") or []
    del max_evidence
    evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
    system = (
        "You are a gold-aware evidence verifier for a memory QA benchmark. "
        "Given the user question, gold answer, and retrieved evidence, judge whether "
        "the evidence alone is sufficient for a separate answer model to derive the "
        "gold answer. Do not require exact wording. Penalize missing comparison sides, "
        "missing list entities, unsupported temporal claims, off-topic evidence, and "
        "missing visual details. Empty evidence is not sufficient, including for "
        "not-mentioned or absence answers. Return only valid JSON."
    )
    user = f"""Question:
{row.get('question')}

Gold answer:
{row.get('gold_answer')}

Retrieved evidence:
{evidence_text}

Return JSON with keys:
{{
  "answerable": boolean,
  "score": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "reason": "short diagnostic"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_judge_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        value = json.loads(match.group(0)) if match else {}
    correct = bool(value.get("correct")) if isinstance(value, dict) else False
    score = float(value.get("score", 1.0 if correct else 0.0)) if isinstance(value, dict) else 0.0
    return {
        "judge_correct": correct,
        "judge_score": 1.0 if score >= 0.5 else 0.0,
        "judge_reason": str(value.get("reason", "")) if isinstance(value, dict) else "",
    }


def _parse_evidence_answerable_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        value = json.loads(match.group(0)) if match else {}
    if not isinstance(value, dict):
        value = {}

    def _float_key(key: str, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value.get(key, default))))
        except (TypeError, ValueError):
            return default

    relevance = _float_key("relevance")
    completeness = _float_key("completeness")
    score = _float_key("score", 1.0 if bool(value.get("answerable")) else 0.0)
    answerable = bool(value.get("answerable"))
    return {
        "judge_correct": answerable,
        "judge_score": 1.0 if answerable else 0.0,
        "judge_reason": str(value.get("reason", "")),
        "evidence_answerable": answerable,
        "evidence_answerable_score": score,
        "evidence_relevance": relevance,
        "evidence_completeness": completeness,
    }


def run_judge(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    llm = LLM(
        model=args.judge_model,
        trust_remote_code=True,
        tensor_parallel_size=args.judge_tp,
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        max_model_len=args.judge_max_model_len,
        dtype="bfloat16",
    )
    if args.judge_mode == "evidence_answerable":
        prompt_builder = lambda row: _evidence_answerable_prompt(row, args.answer_max_evidence)
        parse_judge = _parse_evidence_answerable_json
    else:
        prompt_builder = _judge_prompt
        parse_judge = _parse_judge_json
    prompts = [
        tokenizer.apply_chat_template(
            prompt_builder(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.judge_max_new_tokens)
    generations = llm.generate(prompts, sampling, use_tqdm=True)

    judged: list[dict[str, Any]] = []
    for row, generation in zip(rows, generations, strict=True):
        text = generation.outputs[0].text
        try:
            parsed = parse_judge(text)
        except Exception as exc:
            parsed = {"judge_correct": False, "judge_score": 0.0, "judge_reason": f"judge_parse_error: {exc}"}
        judged.append({**row, **parsed, "judge_raw": text, "judge_mode": args.judge_mode})

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return judged


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}

    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str(row.get("point") or "")].append(row)
        by_scenario[str(row.get("scenario") or "")].append(row)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "num_samples": n,
        "judge_mode": next((str(row.get("judge_mode")) for row in rows if row.get("judge_mode")), ""),
        "judge_accuracy": mean([float(row.get("judge_score") or 0.0) for row in rows]),
        "evidence_answerable_rate": mean([float(row.get("evidence_answerable") or 0.0) for row in rows]),
        "evidence_answerable_score": mean([float(row.get("evidence_answerable_score") or 0.0) for row in rows]),
        "evidence_relevance": mean([float(row.get("evidence_relevance") or 0.0) for row in rows]),
        "evidence_completeness": mean([float(row.get("evidence_completeness") or 0.0) for row in rows]),
        "answer_contains_gold_rate": mean([float(row.get("answer_contains_gold") or 0.0) for row in rows]),
        "evidence_present_rate": mean([1.0 if int(row.get("evidence_count") or 0) > 0 else 0.0 for row in rows]),
        "avg_evidence_count": mean([float(row.get("evidence_count") or 0.0) for row in rows]),
        "avg_num_turns": mean([float(row.get("num_turns") or 0.0) for row in rows]),
        "error_count": sum(1 for row in rows if row.get("error")),
        "answer_error_count": sum(1 for row in rows if row.get("answer_model_error")),
        "counts_by_point": dict(sorted(Counter(str(row.get("point") or "") for row in rows).items())),
        "judge_accuracy_by_point": {
            key: mean([float(row.get("judge_score") or 0.0) for row in value])
            for key, value in sorted(by_point.items())
        },
        "judge_accuracy_by_scenario": {
            key: mean([float(row.get("judge_score") or 0.0) for row in value])
            for key, value in sorted(by_scenario.items())
        },
    }


def write_outputs(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--qas-path", default=DEFAULT_QAS)
    parser.add_argument("--train-sample-ids", default=DEFAULT_TRAIN_IDS)
    parser.add_argument("--train-rlhf-path", default=DEFAULT_TRAIN_RLHF)
    parser.add_argument("--input", default="", help="Existing rollout JSONL to read when --skip-rollout is set.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--tool-format", default="qwen3_coder")
    parser.add_argument("--student-tp", type=int, default=1)
    parser.add_argument("--student-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--judge-tp", type=int, default=2)
    parser.add_argument("--judge-mode", choices=["answer_correctness", "evidence_answerable"], default="answer_correctness")
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--judge-max-model-len", type=int, default=16384)
    parser.add_argument("--judge-max-new-tokens", type=int, default=192)
    parser.add_argument("--answer-base-url", default=DEFAULT_ANSWER_BASE_URL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-timeout", type=float, default=60.0)
    parser.add_argument("--answer-max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-max-evidence", type=int, default=12)
    parser.add_argument("--skip-answer-model", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_rollout:
        input_path = Path(args.input or args.output)
        rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        qas = load_eval_qas(args)
        rows = rollout_student(args, qas)
        write_outputs(rows, args.output)
    if not args.rollout_only and not args.skip_answer_model and args.judge_mode != "evidence_answerable":
        rows = run_answer_model(args, rows)
        write_outputs(rows, args.output)
    if not args.rollout_only:
        rows = run_judge(args, rows)
        write_outputs(rows, args.output)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
