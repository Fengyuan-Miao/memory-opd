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

"""Executor for validated OPD-MM tool trajectories."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Protocol

from .models import (
    EvidenceItem,
    ExecutionResult,
    ExecutionStep,
    PoolItem,
    ToolAction,
)
from .retrieval import HiddenMemoryStore, HybridRetriever
from .schema import TrajectoryValidator

PUBLIC_EVIDENCE_FIELDS = (
    "content",
    "timestamp",
    "session_date",
    "author",
    "modality",
)
TIMESTAMP_DATE_PATTERN = re.compile(
    r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
)
DATE_ONLY_PATTERN = re.compile(
    r"^\s*(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\s*$"
)


class RawInspector(Protocol):
    def inspect(
        self,
        image_path: str,
        query: str,
        question_image: Optional[str] = None,
        text_context: Optional[str] = None,
    ) -> str:
        ...


class ToolExecutor:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        raw_inspector: Optional[RawInspector] = None,
        validator: Optional[TrajectoryValidator] = None,
        max_raw_inspections: int = 3,
    ):
        self.retriever = retriever or HybridRetriever()
        self.raw_inspector = raw_inspector
        self.validator = validator or TrajectoryValidator()
        self.max_raw_inspections = max(0, int(max_raw_inspections))

    def run(
        self,
        trace: List[ToolAction] | List[dict[str, Any]],
        query: str,
        memory_store: HiddenMemoryStore,
        question_image: Optional[str] = None,
    ) -> ExecutionResult:
        actions = self.validator.validate(trace)
        pool = memory_store.initial_pool()
        pool_has_candidates = False
        evidence: List[EvidenceItem] = []
        steps: List[ExecutionStep] = []
        stopped = False
        raw_calls = 0
        error = ""

        for index, action in enumerate(actions):
            before = len(pool)
            step_error = ""
            evidence_added = 0
            try:
                if action.tool == "FILTER":
                    source_pool = self._filter_source_pool(
                        pool,
                        memory_store,
                        action.arguments["scope"],
                    )
                    filtered = self._filter(
                        source_pool,
                        field=action.arguments["field"],
                        op=action.arguments["op"],
                        value=action.arguments["value"],
                    )
                    pool = filtered
                    pool_has_candidates = True
                    evidence_added = len(self._refresh_evidence_from_pool(evidence, pool, source="FILTER"))
                elif action.tool == "SORT":
                    pool = self._sort(pool, **action.arguments)
                    if pool_has_candidates:
                        evidence_added = len(self._refresh_evidence_from_pool(evidence, pool, source="SORT"))
                elif action.tool == "TOPK":
                    pool = self._topk_turns(pool, action.arguments["k"])
                    if pool_has_candidates:
                        evidence_added = len(self._refresh_evidence_from_pool(evidence, pool, source="TOPK"))
                elif action.tool == "RETRIEVE":
                    retrieve_query = action.arguments.get("query") or query
                    retrieved = self.retriever.retrieve(
                        memory_store.initial_pool(),
                        query=retrieve_query,
                        store=memory_store,
                        method=action.arguments.get("method", "hybrid"),
                        top_k=action.arguments.get("top_k", 5),
                        question_image=question_image,
                    )
                    pool = retrieved
                    pool_has_candidates = True
                    evidence_added = len(self._refresh_evidence_from_pool(evidence, pool, source="RETRIEVE"))
                elif action.tool == "EXPAND_NEIGHBORS":
                    if not pool_has_candidates or not pool:
                        raise ValueError("EXPAND_NEIGHBORS requires an existing candidate pool")
                    expanded = self._expand_neighbors(
                        pool,
                        memory_store,
                        action.arguments["window"],
                    )
                    pool = self._merge_pools(pool, expanded)
                    pool_has_candidates = True
                    evidence_added = len(self._refresh_evidence_from_pool(evidence, pool, source="EXPAND_NEIGHBORS"))
                elif action.tool == "INSPECT_RAW":
                    remaining = max(0, self.max_raw_inspections - raw_calls)
                    inspected = self._inspect_raw(
                        pool,
                        query,
                        remaining,
                        question_image=question_image,
                    )
                    raw_calls += len(inspected)
                    evidence.extend(inspected)
                    evidence_added = len(inspected)
                elif action.tool == "STOP":
                    stopped = True
            except Exception as exc:
                step_error = str(exc)
                error = f"action {index} {action.tool}: {exc}"
            steps.append(
                ExecutionStep(
                    index=index,
                    action=action,
                    pool_before=before,
                    pool_after=len(pool),
                    evidence_added=evidence_added,
                    error=step_error,
                )
            )
            if stopped or step_error:
                break
        if not evidence and not error:
            evidence = self._pool_evidence(pool, source="FINAL_POOL")

        return ExecutionResult(
            evidence=evidence,
            steps=steps,
            final_pool_size=len(pool),
            final_memory_ids=[item.memory.memory_id for item in pool],
            stopped=stopped,
            error=error,
            raw_inspection_calls=raw_calls,
        )

    @staticmethod
    def _merge_pools(existing: List[PoolItem], incoming: List[PoolItem]) -> List[PoolItem]:
        """Merge candidate pools by hidden memory id while preserving stable order.

        If an incoming item already exists, keep its position but refresh score
        and retrieved status so later INSPECT_RAW can inspect records that were
        first introduced by FILTER and then selected by RETRIEVE.
        """
        merged = list(existing)
        positions = {item.memory.memory_id: index for index, item in enumerate(merged)}
        for item in incoming:
            memory_id = item.memory.memory_id
            if memory_id in positions:
                index = positions[memory_id]
                previous = merged[index]
                merged[index] = PoolItem(
                    memory=item.memory,
                    score=item.score or previous.score,
                    retrieved=previous.retrieved or item.retrieved,
                )
                continue
            positions[memory_id] = len(merged)
            merged.append(item)
        return merged

    @staticmethod
    def _topk_turns(pool: List[PoolItem], k: int) -> List[PoolItem]:
        selected_turns = []
        selected = []
        for item in pool:
            turn_id = item.memory.turn_id
            if turn_id not in selected_turns:
                if len(selected_turns) >= k:
                    continue
                selected_turns.append(turn_id)
            selected.append(item)
        return selected

    @staticmethod
    def _expand_neighbors(
        pool: List[PoolItem],
        memory_store: HiddenMemoryStore,
        window: int,
    ) -> List[PoolItem]:
        """Return records from turns neighboring the current candidate pool.

        The unit of expansion is a dialogue turn identified by session plus a
        turn index. Mem-Gallery records may store the index as ``turn_index``,
        ``round_id`` or a trailing numeric component in ``turn_id``. Records
        lacking a structured position cannot contribute neighbors, but existing
        records in the selected pool are still preserved by the caller's merge.
        """
        selected_keys = {
            key
            for item in pool
            if (key := ToolExecutor._memory_session_turn_key(item.memory)) is not None
        }
        expanded_keys = set(selected_keys)
        for session_id, turn_index in selected_keys:
            for distance in range(1, int(window) + 1):
                expanded_keys.add((session_id, turn_index - distance))
                expanded_keys.add((session_id, turn_index + distance))

        score_by_turn = {item.memory.turn_id: item.score for item in pool}
        expanded = []
        for item in memory_store.initial_pool():
            key = ToolExecutor._memory_session_turn_key(item.memory)
            if key not in expanded_keys:
                continue
            expanded.append(
                PoolItem(
                    item.memory,
                    score_by_turn.get(item.memory.turn_id, 0.0),
                    retrieved=item.retrieved,
                )
            )

        def sort_key(item: PoolItem) -> tuple[Any, ...]:
            key = ToolExecutor._memory_session_turn_key(item.memory) or ("", 0)
            return key[0], key[1], item.memory.timestamp, item.memory.memory_id

        expanded.sort(key=sort_key)
        return expanded

    @staticmethod
    def _memory_session_turn_key(memory: Any) -> tuple[str, int] | None:
        metadata = getattr(memory, "metadata", {}) or {}
        session_id = metadata.get("session_id") or metadata.get("dialogue_id") or metadata.get("conversation_id")
        if not session_id:
            turn_id = str(getattr(memory, "turn_id", "") or "")
            parts = turn_id.split(":")
            if len(parts) >= 2:
                session_id = parts[-2]
        if not session_id:
            return None

        scenario = str(metadata.get("scenario") or "").strip()
        session_key = f"{scenario}:{session_id}" if scenario else str(session_id)
        turn_index = None
        for candidate in (
            metadata.get("turn_index"),
            metadata.get("round_index"),
            metadata.get("turn_number"),
            metadata.get("round_id"),
            getattr(memory, "turn_id", ""),
        ):
            if candidate is None or str(candidate).strip() == "":
                continue
            turn_index = ToolExecutor._coerce_turn_index(candidate)
            if turn_index is not None:
                break
        if turn_index is None:
            return None
        return session_key, turn_index

    @staticmethod
    def _coerce_turn_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        text = str(value or "").strip()
        if text.isdigit():
            return int(text)
        match = re.search(r"(?:^|[:_-])(\d+)$", text)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _filter_source_pool(
        pool: List[PoolItem],
        memory_store: HiddenMemoryStore,
        scope: str = "current_pool",
    ) -> List[PoolItem]:
        if scope == "full_memory":
            return memory_store.initial_pool()
        return pool

    @staticmethod
    def _filter(
        pool: List[PoolItem],
        field: str,
        op: str,
        value: Any,
    ) -> List[PoolItem]:
        target = str(value).lower()

        def keep(item: PoolItem) -> bool:
            current_value = item.memory.field_value(field)
            if field == "timestamp":
                return ToolExecutor._match_timestamp_filter(current_value, op, value)
            current = str(current_value or "").lower()
            if op == "eq":
                return current == target
            if op == "neq":
                return current != target
            if op == "contains":
                return target in current
            if op == "before":
                return current < target
            if op == "after":
                return current > target
            return False

        return [item for item in pool if keep(item)]

    @classmethod
    def _match_timestamp_filter(cls, current_value: Any, op: str, target_value: Any) -> bool:
        """Match timestamp filters with date-only model outputs.

        Mem-Gallery records store timestamps as values like ``2024-06-17T0004``
        while models naturally emit date-only filters such as ``2024-06-17``.
        For timestamp fields, date-only equality/contains therefore matches the
        record date prefix. before/after compare dates when the target omits a
        time/turn suffix, and fall back to normalized timestamp comparison when
        a more specific target is provided.
        """
        current = str(current_value or "")
        target = str(target_value or "")
        current_lower = current.lower()
        target_lower = target.lower()
        current_date = cls._canonical_date(current)
        target_date = cls._canonical_date(target)
        target_is_date_only = cls._is_date_only(target)

        if op == "eq":
            if current_date and target_date:
                if target_is_date_only:
                    return current_date == target_date
                return cls._normalized_timestamp(current) == cls._normalized_timestamp(target)
            return current_lower == target_lower
        if op == "neq":
            return not cls._match_timestamp_filter(current_value, "eq", target_value)
        if op == "contains":
            if target_lower in current_lower:
                return True
            return bool(current_date and target_date and current_date == target_date)
        if op in {"before", "after"}:
            current_key = current_date if target_is_date_only else cls._normalized_timestamp(current)
            target_key = target_date if target_is_date_only else cls._normalized_timestamp(target)
            if not current_key or not target_key:
                current_key = current_lower
                target_key = target_lower
            return current_key < target_key if op == "before" else current_key > target_key
        return False

    @staticmethod
    def _canonical_date(value: Any) -> str:
        match = TIMESTAMP_DATE_PATTERN.search(str(value or ""))
        if not match:
            return ""
        return (
            f"{int(match.group('year')):04d}-"
            f"{int(match.group('month')):02d}-"
            f"{int(match.group('day')):02d}"
        )

    @staticmethod
    def _is_date_only(value: Any) -> bool:
        return DATE_ONLY_PATTERN.match(str(value or "")) is not None

    @classmethod
    def _normalized_timestamp(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        match = TIMESTAMP_DATE_PATTERN.search(text)
        if not match:
            return text
        suffix = text[match.end() :].strip()
        if suffix and suffix[0].isdigit():
            suffix = f"t{suffix}"
        return f"{cls._canonical_date(text)}{suffix}"

    @classmethod
    def _sort(
        cls,
        pool: List[PoolItem],
        field: str,
        order: str,
    ) -> List[PoolItem]:
        reverse = order == "desc"
        if field == "score":
            key = lambda item: item.score
        elif field == "turn_id":
            key = lambda item: cls._natural_key(item.memory.turn_id)
        else:
            key = lambda item: str(item.memory.field_value(field) or "")
        return sorted(pool, key=key, reverse=reverse)

    @staticmethod
    def _natural_key(value: str) -> tuple[Any, ...]:
        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value or ""))
        )

    @staticmethod
    def _pool_evidence(
        pool: List[PoolItem],
        fields: tuple[str, ...] = PUBLIC_EVIDENCE_FIELDS,
        source: str = "FINAL_POOL",
    ) -> List[EvidenceItem]:
        evidence = []
        for item in pool:
            values = {}
            for field in fields:
                if field == "content":
                    # Public observations expose one answer-useful text field.
                    # Some image records store their caption in summary only,
                    # so fall back to it without duplicating the same text.
                    values[field] = item.memory.content or item.memory.summary
                else:
                    values[field] = item.memory.field_value(field)
            if "session_date" not in values:
                values["session_date"] = item.memory.metadata.get(
                    "session_date"
                )
            image_id = item.memory.public_image_id()
            if image_id:
                values["image_id"] = image_id
            if item.score:
                values["retrieval_score"] = item.score
            values = {key: value for key, value in values.items() if value not in (None, "")}
            evidence.append(
                EvidenceItem(
                    memory_id=item.memory.memory_id,
                    fields=values,
                    source=source,
                )
            )
        return evidence

    @classmethod
    def _append_pool_evidence(
        cls,
        evidence: List[EvidenceItem],
        pool: List[PoolItem],
        source: str,
    ) -> List[EvidenceItem]:
        """Append public pool records as evidence without duplicating memories."""
        existing = {item.memory_id for item in evidence}
        added = []
        for item in cls._pool_evidence(pool, source=source):
            if item.memory_id in existing:
                continue
            evidence.append(item)
            added.append(item)
            existing.add(item.memory_id)
        return added

    @classmethod
    def _refresh_evidence_from_pool(
        cls,
        evidence: List[EvidenceItem],
        pool: List[PoolItem],
        source: str,
    ) -> List[EvidenceItem]:
        """Refresh answer evidence from the current candidate pool.

        Pool-mutating tools should expose the latest candidate view instead of
        an ever-growing union of stale candidates. Preserve raw visual
        inspection evidence only for memories that remain in the current pool.
        """
        pool_ids = {item.memory.memory_id for item in pool}
        preserved_raw = [
            item
            for item in evidence
            if item.source == "INSPECT_RAW" and item.memory_id in pool_ids
        ]
        old_signatures = {
            (item.memory_id, item.source, tuple(sorted(item.fields.items())))
            for item in evidence
        }
        del source
        refreshed = cls._pool_evidence(pool, source="MEMORY")
        existing_ids = {item.memory_id for item in refreshed}
        for item in preserved_raw:
            if item.memory_id in existing_ids:
                refreshed.append(item)
        evidence[:] = refreshed
        return [
            item
            for item in refreshed
            if (item.memory_id, item.source, tuple(sorted(item.fields.items())))
            not in old_signatures
        ]

    def _inspect_raw(
        self,
        pool: List[PoolItem],
        query: str,
        limit: int,
        question_image: Optional[str] = None,
    ) -> List[EvidenceItem]:
        if self.raw_inspector is None:
            return []
        inspect_pool = [item for item in pool if item.retrieved]
        if not inspect_pool:
            return []
        evidence = []
        text_by_turn = self._text_context_by_turn(inspect_pool)
        for item in inspect_pool:
            if len(evidence) >= limit:
                break
            pointer = item.memory.raw_pointer
            if not pointer:
                continue
            context = text_by_turn.get(item.memory.turn_id, "")
            observation = self.raw_inspector.inspect(
                pointer,
                query,
                question_image=question_image,
                text_context=context,
            )
            fields = {
                "visual_observation": observation,
                "linked_text_context": context,
                "image_label": f"context={context[:220]}",
                "session_date": item.memory.metadata.get(
                    "session_date"
                ),
                "timestamp": item.memory.timestamp,
            }
            image_id = item.memory.public_image_id()
            if image_id:
                fields["image_id"] = image_id
            evidence.append(
                EvidenceItem(
                    memory_id=item.memory.memory_id,
                    fields=fields,
                    source="INSPECT_RAW",
                )
            )
        return evidence

    @staticmethod
    def _text_context_by_turn(pool: List[PoolItem]) -> dict[str, str]:
        contexts: dict[str, List[str]] = {}
        for item in pool:
            memory = item.memory
            if memory.raw_pointer:
                continue
            text = " ".join(
                value
                for value in [memory.summary, memory.content]
                if value
            ).strip()
            if not text:
                continue
            contexts.setdefault(memory.turn_id, []).append(text)
        return {
            turn_id: " ".join(values)[:1200]
            for turn_id, values in contexts.items()
        }
