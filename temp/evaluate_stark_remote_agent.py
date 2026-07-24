#!/usr/bin/env python3
"""Run a fixed STARK OPD-MM sample through a remote OpenAI-compatible agent."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import aiohttp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.dataset import opd_messages_for_state
from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.models import ToolAction
from verl.experimental.opd_mm.retrieval import TurnAwareHybridRetriever
from verl.experimental.opd_mm.schema import TrajectoryValidator
from verl.experimental.opd_mm.tools import (
    OPDToolSession,
    hidden_store_from_records,
    openai_tool_schemas,
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, tuple) else []


def _data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text") or "").strip()
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(value or "").strip()


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        value = json.loads(match.group(0)) if match else {}
    return value if isinstance(value, dict) else {}


class DirectRawInspector:
    """Synchronous inspector used by ToolExecutor, with Qwen thinking disabled."""

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.url = base_url.rstrip("/") + (
            "/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions"
        )
        self.model = model
        self.timeout = timeout
        self.opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))

    def inspect(
        self,
        image_path: str,
        query: str,
        question_image: str | None = None,
        text_context: str | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Inspect the memory image and return only concise facts relevant to the question.\n"
                    f"Question: {query}\n"
                    f"Same-turn text: {(text_context or '').strip() or '(none)'}"
                ),
            },
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
        ]
        if question_image and Path(question_image).exists():
            content.extend(
                [
                    {"type": "text", "text": "Held-out question image for comparison:"},
                    {"type": "image_url", "image_url": {"url": _data_url(question_image)}},
                ]
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a raw visual inspector. Report visible query-relevant facts only. "
                        "Do not expose hidden memory identifiers."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib_request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            return _text_content(parsed["choices"][0]["message"].get("content"))
        except Exception as exc:
            return f"RAW_INSPECT_ERROR: {type(exc).__name__}: {exc}"


class RemoteClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        timeout: float,
    ):
        self.url = base_url.rstrip("/") + (
            "/chat/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/chat/completions"
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "RemoteClient":
        self.session = aiohttp.ClientSession(timeout=self.timeout, trust_env=False)
        return self

    async def __aexit__(self, *_args: Any) -> None:
        assert self.session is not None
        await self.session.close()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 4,
    ) -> dict[str, Any]:
        assert self.session is not None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            # qwen3_coder auto parsing works here. "required" asks vLLM to
            # compile the schema and currently rejects JSON Schema uniqueItems.
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                async with self.semaphore:
                    async with self.session.post(self.url, json=payload) as response:
                        raw = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"HTTP {response.status}: {raw[:800]}")
                        parsed = json.loads(raw)
                        if parsed.get("error"):
                            raise RuntimeError(str(parsed["error"]))
                        return parsed
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"remote model request failed after {retries} attempts: {last_error}")


def _base_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [dict(item) for item in _as_list(row["prompt"])]
    question_image = row["extra_info"].get("question_image")
    if question_image and Path(str(question_image)).exists():
        for message in reversed(messages):
            if message.get("role") == "user":
                text = str(message.get("content") or "")
                message["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": _data_url(question_image)}},
                ]
                break
    return messages


def _make_session(
    row: dict[str, Any],
    raw_inspector: DirectRawInspector,
    vector_store_dir: str,
    vector_device: str,
) -> OPDToolSession:
    runtime = dict(row["extra_info"]["tools_kwargs"]["opd_mm"])
    records = _as_list(runtime["records"])
    return OPDToolSession(
        executor=ToolExecutor(
            retriever=TurnAwareHybridRetriever(),
            raw_inspector=raw_inspector,
            validator=TrajectoryValidator(
                max_actions=int(runtime.get("max_actions", 10)),
                max_top_k=50,
                allow_inspect_raw=bool(runtime.get("allow_inspect_raw", True)),
            ),
            max_raw_inspections=int(runtime.get("max_raw_inspections", 3)),
            max_pool_size=int(runtime.get("max_pool_size", 24)),
        ),
        memory_store=hidden_store_from_records(
            records,
            vector_store_dir=vector_store_dir,
            vector_device=vector_device,
        ),
        query=str(runtime.get("query") or row["extra_info"].get("question") or ""),
        question_image=runtime.get("question_image"),
    )


async def _rollout(
    client: RemoteClient,
    row: dict[str, Any],
    raw_inspector: DirectRawInspector,
    tool_schemas: list[dict[str, Any]],
    vector_store_dir: str,
    vector_device: str,
) -> dict[str, Any]:
    extra = row["extra_info"]
    base_messages = _base_messages(row)
    messages = base_messages
    session = _make_session(
        row,
        raw_inspector,
        vector_store_dir,
        vector_device,
    )
    generations: list[dict[str, Any]] = []
    model_error = ""

    try:
        for _ in range(session.executor.validator.max_actions):
            response = await client.chat(messages, tools=tool_schemas, max_tokens=384)
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            generations.append(
                {
                    "content": _text_content(message.get("content")),
                    "tool_calls": tool_calls,
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "usage": response.get("usage"),
                }
            )
            if not tool_calls:
                model_error = "no_tool_call"
                break
            function = tool_calls[0].get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                action = ToolAction(str(function.get("name") or "").upper(), arguments)
                observation = await asyncio.to_thread(session.execute, action)
            except Exception as exc:
                model_error = f"tool_call_error: {type(exc).__name__}: {exc}"
                break
            messages = opd_messages_for_state(
                base_messages,
                [item.to_dict() for item in session.trace],
                observation,
            )
            if session.stopped or observation.get("error"):
                break
    except Exception as exc:
        model_error = f"rollout_error: {type(exc).__name__}: {exc}"

    state = session.public_state()
    return {
        "sample_id": str(extra["sample_id"]),
        "scenario": str(extra["scenario"]),
        "point": str(extra["point"]),
        "question": str(session.query),
        "question_image": extra.get("question_image"),
        "gold_answer": str(extra["gold_answer"]),
        "trace": state["trace"],
        "evidence": state["evidence"],
        "evidence_count": state["evidence_count"],
        "evidence_memory_count": state["evidence_memory_count"],
        "pool_count": state["pool_count"],
        "num_turns": len(state["trace"]),
        "stopped": state["stopped"],
        "max_actions_reached": state["max_actions_reached"],
        "drop_calls": state["drop_calls"],
        "dropped_evidence_count": state["dropped_evidence_count"],
        "raw_inspection_calls": state["raw_inspection_calls"],
        "session_error": state["error"],
        "model_error": model_error,
        "raw_generations": generations,
        "vector_store_dir": vector_store_dir,
    }


def _answer_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Question:\n"
                f"{row['question']}\n\n"
                "Retrieved evidence:\n"
                f"{json.dumps(row['evidence'], ensure_ascii=False, indent=2)}\n\n"
                "Give the final answer only."
            ),
        }
    ]
    question_image = row.get("question_image")
    if question_image and Path(str(question_image)).exists():
        content.append({"type": "image_url", "image_url": {"url": _data_url(question_image)}})
    return [
        {
            "role": "system",
            "content": (
                "Answer the memory-QA question using only the retrieved evidence and attached question image, "
                "if present. Do not mention tool calls or the gold answer. Give a concise final answer."
            ),
        },
        {"role": "user", "content": content},
    ]


def _judge_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge answer correctness for a memory-QA benchmark. Use the gold answer as the sole reference. "
                "The candidate is correct only when it directly answers the question and is semantically "
                "equivalent to the gold answer. Return only JSON with keys correct (boolean), score (0 or 1), "
                "and reason (short string)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{row['question']}\n\n"
                f"Gold answer:\n{row['gold_answer']}\n\n"
                f"Candidate answer:\n{row.get('answer_model_answer', '')}\n\n"
                "Is the candidate semantically correct?"
            ),
        },
    ]


async def _answer_and_judge(client: RemoteClient, row: dict[str, Any]) -> dict[str, Any]:
    try:
        answer_response = await client.chat(_answer_messages(row), max_tokens=256)
        answer = _text_content(answer_response["choices"][0]["message"].get("content"))
        answer_error = "" if answer else "empty_answer"
    except Exception as exc:
        answer = ""
        answer_error = f"{type(exc).__name__}: {exc}"
    judged = False
    score = 0.0
    reason = ""
    judge_raw = ""
    judge_error = ""
    try:
        judge_row = {**row, "answer_model_answer": answer}
        judge_response = await client.chat(_judge_messages(judge_row), max_tokens=192)
        judge_raw = _text_content(judge_response["choices"][0]["message"].get("content"))
        try:
            value = _json_object(judge_raw)
            if not isinstance(value.get("correct"), bool):
                raise ValueError("judge JSON has no boolean correct field")
            judged = bool(value["correct"])
            score = 1.0 if judged else 0.0
            reason = str(value.get("reason") or "")
        except Exception:
            # Recover only a single unambiguous JSON-style boolean when the
            # model truncates the reason or final closing brace.
            matches = re.findall(
                r'["\']correct["\']\s*:\s*(true|false)\b',
                judge_raw,
                flags=re.IGNORECASE,
            )
            verdicts = {match.lower() == "true" for match in matches}
            if len(verdicts) != 1:
                raise
            judged = verdicts.pop()
            score = 1.0 if judged else 0.0
            reason = ""
    except Exception as exc:
        judge_error = f"{type(exc).__name__}: {exc}"
    return {
        **row,
        "answer_model_answer": answer,
        "answer_model_error": answer_error,
        "judge_correct": judged,
        "judge_score": score,
        "judge_reason": reason,
        "judge_raw": judge_raw,
        "judge_error": judge_error,
    }


def _sample_rows(path: Path, max_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = pd.read_parquet(path).to_dict("records")
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str(row["extra_info"]["point"])].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    # Keep the sole reviewed TTL test item in the fixed 100; fill the rest
    # proportionally by random round-robin over all non-empty categories.
    for row in by_point.pop("TTL", []):
        selected.append(row)
    for values in by_point.values():
        rng.shuffle(values)
    keys = sorted(by_point)
    rng.shuffle(keys)
    while len(selected) < max_samples and keys:
        next_keys = []
        for key in keys:
            values = by_point[key]
            if values:
                selected.append(values.pop())
                if values:
                    next_keys.append(key)
            if len(selected) >= max_samples:
                break
        keys = next_keys
    return sorted(selected, key=lambda row: str(row["extra_info"]["sample_id"]))


def _summary(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    n = len(rows)
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_counts: Counter[str] = Counter()
    retrieve_methods: Counter[str] = Counter()
    retrieve_top_k: list[int] = []
    for row in rows:
        by_point[row["point"]].append(row)
        for action in row["trace"]:
            tool = str(action.get("tool") or "")
            action_counts[tool] += 1
            if tool == "RETRIEVE":
                retrieve_methods[str(action.get("method") or "")] += 1
                if isinstance(action.get("top_k"), int):
                    retrieve_top_k.append(action["top_k"])

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "num_samples": n,
        "answer_correct_accuracy": mean([float(row["judge_score"]) for row in rows]),
        "accuracy_by_point": {
            point: mean([float(row["judge_score"]) for row in values])
            for point, values in sorted(by_point.items())
        },
        "counts_by_point": dict(sorted(Counter(row["point"] for row in rows).items())),
        "evidence_present_rate": mean([1.0 if row["evidence_count"] else 0.0 for row in rows]),
        "avg_evidence_count": mean([float(row["evidence_count"]) for row in rows]),
        "avg_evidence_memory_count": mean([float(row["evidence_memory_count"]) for row in rows]),
        "avg_num_turns": mean([float(row["num_turns"]) for row in rows]),
        "stop_rate": mean([1.0 if row["stopped"] else 0.0 for row in rows]),
        "max_actions_reached_rate": mean([1.0 if row["max_actions_reached"] else 0.0 for row in rows]),
        "drop_trajectory_rate": mean([1.0 if row["drop_calls"] else 0.0 for row in rows]),
        "avg_dropped_evidence": mean([float(row["dropped_evidence_count"]) for row in rows]),
        "raw_inspection_trajectory_rate": mean([1.0 if row["raw_inspection_calls"] else 0.0 for row in rows]),
        "action_counts": dict(sorted(action_counts.items())),
        "retrieve_method_counts": dict(sorted(retrieve_methods.items())),
        "avg_retrieve_top_k": mean([float(value) for value in retrieve_top_k]),
        "model_error_count": sum(bool(row["model_error"]) for row in rows),
        "session_error_count": sum(bool(row["session_error"]) for row in rows),
        "answer_error_count": sum(bool(row["answer_model_error"]) for row in rows),
        "judge_error_count": sum(bool(row["judge_error"]) for row in rows),
        "vector_index_enabled": all(bool(row.get("vector_store_dir")) for row in rows),
        "vector_store_dir": next(
            (str(row.get("vector_store_dir")) for row in rows if row.get("vector_store_dir")),
            "",
        ),
        "elapsed_seconds": elapsed,
    }


async def main_async(args: argparse.Namespace) -> None:
    rows = _sample_rows(Path(args.rlhf_path), args.max_samples, args.seed)
    if args.smoke:
        rows = rows[: args.smoke]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tool_schemas = openai_tool_schemas(include_inspect_raw=True)
    raw_inspector = DirectRawInspector(args.base_url, args.model, args.timeout)
    started = time.time()

    # Populate the shared query-encoder caches before concurrent trajectories
    # can race to load the same models more than once.
    first_runtime = dict(rows[0]["extra_info"]["tools_kwargs"]["opd_mm"])
    warm_store = hidden_store_from_records(
        _as_list(first_runtime["records"]),
        vector_store_dir=args.vector_store_dir,
        vector_device=args.vector_device,
    )
    warm_store.query_vectors("OPD-MM index warmup")
    warm_store.vision_query_vector("OPD-MM index warmup")
    warm_store.hybrid_query_vector("OPD-MM index warmup")

    async with RemoteClient(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
    ) as client:
        rollout_tasks = [
            asyncio.create_task(
                _rollout(
                    client,
                    row,
                    raw_inspector,
                    tool_schemas,
                    args.vector_store_dir,
                    args.vector_device,
                )
            )
            for row in rows
        ]
        rollouts: list[dict[str, Any]] = []
        for completed, task in enumerate(asyncio.as_completed(rollout_tasks), start=1):
            rollouts.append(await task)
            if completed % 10 == 0 or completed == len(rollout_tasks):
                print(f"rollout {completed}/{len(rollout_tasks)}", flush=True)
        judge_tasks = [asyncio.create_task(_answer_and_judge(client, row)) for row in rollouts]
        results: list[dict[str, Any]] = []
        for completed, task in enumerate(asyncio.as_completed(judge_tasks), start=1):
            results.append(await task)
            if completed % 10 == 0 or completed == len(judge_tasks):
                print(f"answer+judge {completed}/{len(judge_tasks)}", flush=True)

    results.sort(key=lambda row: row["sample_id"])
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = _summary(results, time.time() - started)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rlhf-path",
        default=(
            "dataset/Stark/opd_mm_store_rounds_3000/expansion_v3/"
            "direct_api_3000_rounds_clean_qwen35_9b_support_ablation_"
            "cd_recleaned_full365_ttl_reviewed/rlhf/test.parquet"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--vector-store-dir",
        default="dataset/Stark/opd_mm_store_rounds_3000",
    )
    parser.add_argument("--vector-device", default="cuda:4")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument(
        "--output",
        default="outputs/opd_mm_eval/stark_ttlreviewed_qwen35_9b_remoteagent_test100_20260723.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
