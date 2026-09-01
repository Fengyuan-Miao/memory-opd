"""Cutoff-aware, event-level minimal evidence mining."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable

from .ledger import ObservedFactLedger
from .models import MemoryCutoff, QACandidate


def _minimal_sets(values: Iterable[frozenset[str]]) -> list[frozenset[str]]:
    unique = sorted(set(values), key=lambda item: (len(item), sorted(item)))
    result: list[frozenset[str]] = []
    for item in unique:
        if any(existing <= item for existing in result):
            continue
        result.append(item)
    return result


@dataclass(frozen=True)
class EvidenceResult:
    required_evidence_sets: tuple[tuple[str, ...], ...]
    required_visual_fact_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    hard_negatives: tuple[str, ...]
    hard_negative_event_ids: tuple[str, ...]
    hard_negative_image_ids: tuple[str, ...]
    clue_count: int
    distinct_evidence_sessions: int
    session_gap: int
    requires_long_term_memory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_evidence_sets": [list(item) for item in self.required_evidence_sets],
            "required_visual_fact_ids": list(self.required_visual_fact_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "hard_negatives": list(self.hard_negatives),
            "hard_negative_event_ids": list(self.hard_negative_event_ids),
            "hard_negative_image_ids": list(self.hard_negative_image_ids),
            "clue_unit": "event",
            "clue_count": self.clue_count,
            "distinct_evidence_sessions": self.distinct_evidence_sessions,
            "session_gap": self.session_gap,
            "requires_long_term_memory": self.requires_long_term_memory,
        }


class EvidenceMiner:
    def __init__(self, ledger: ObservedFactLedger) -> None:
        self.ledger = ledger

    def support_sets_for_facts(self, fact_ids: list[str], cutoff: MemoryCutoff) -> list[frozenset[str]]:
        view = self.ledger.view(cutoff)
        visible = {fact.fact_id: fact for fact in view.facts}
        alternatives: list[list[str]] = []
        for fact_id in fact_ids:
            fact = visible.get(fact_id)
            if fact is None:
                raise ValueError(f"required fact {fact_id!r} is not visible at the QA cutoff")
            events = sorted(
                {
                    provenance.event_id
                    for provenance in fact.observed_provenance
                    if self.ledger.event_is_visible(provenance.event_id, cutoff)
                },
                key=self.ledger.event_order.__getitem__,
            )
            if not events:
                raise ValueError(f"required fact {fact_id!r} has no visible provenance")
            alternatives.append(events)
        if not alternatives:
            return []
        combinations = (frozenset(choice) for choice in itertools.product(*alternatives))
        return _minimal_sets(combinations)

    def support_sets_for_visual_facts(
        self,
        visual_fact_ids: list[str],
        cutoff: MemoryCutoff,
        *,
        query_image_ids: tuple[str, ...] = (),
    ) -> list[frozenset[str]]:
        alternatives: list[list[str]] = []
        for visual_fact_id in visual_fact_ids:
            image_id = self.ledger.visual_fact_to_image.get(visual_fact_id)
            if image_id is None:
                raise ValueError(f"unknown required visual fact {visual_fact_id!r}")
            if image_id in query_image_ids:
                continue
            events = [
                event.event_id
                for event in self.ledger.episode.events
                if image_id in event.image_ids and self.ledger.event_is_visible(event.event_id, cutoff)
            ]
            if not events:
                raise ValueError(f"visual fact {visual_fact_id!r} has no visible memory event")
            alternatives.append(events)
        if not alternatives:
            return []
        return _minimal_sets(frozenset(choice) for choice in itertools.product(*alternatives))

    @staticmethod
    def _merge_support_sets(
        left: list[frozenset[str]],
        right: list[frozenset[str]],
    ) -> list[frozenset[str]]:
        if not left:
            return right
        if not right:
            return left
        return _minimal_sets(first | second for first in left for second in right)

    def _required_fact_ids(self, qa: QACandidate) -> list[str]:
        value = qa.task_oracle.get("required_fact_ids")
        if isinstance(value, list):
            return [str(item) for item in value]
        result = []
        for key in ("fact_id", "old_fact_id", "new_fact_id", "left_fact_id", "right_fact_id"):
            if qa.task_oracle.get(key):
                result.append(str(qa.task_oracle[key]))
        return list(dict.fromkeys(result))

    def mine(self, qa: QACandidate) -> EvidenceResult:
        if qa.required_evidence_sets:
            sets = [frozenset(item) for item in qa.required_evidence_sets]
        else:
            sets = self.support_sets_for_facts(self._required_fact_ids(qa), qa.memory_cutoff)
            visual_sets = self.support_sets_for_visual_facts(
                list(qa.required_visual_fact_ids),
                qa.memory_cutoff,
                query_image_ids=qa.question_image_ids,
            )
            sets = self._merge_support_sets(sets, visual_sets)
        if qa.task == "TR" and not sets:
            if qa.task_oracle.get("kind") == "temporal_relation":
                temporal_events = [
                    qa.task_oracle.get("left_event_id"),
                    qa.task_oracle.get("right_event_id"),
                ]
            else:
                temporal_events = qa.task_oracle.get("event_ids", [])
            if temporal_events and all(event_id for event_id in temporal_events):
                sets = [frozenset(str(event_id) for event_id in temporal_events)]
        if qa.task == "AR" and not sets:
            anchors = qa.task_oracle.get("topic_anchor_event_ids", [])
            if isinstance(anchors, list) and anchors:
                sets = [frozenset(str(item) for item in anchors)]
        sets = _minimal_sets(sets)
        ordered_sets = [
            tuple(sorted(item, key=self.ledger.event_order.__getitem__))
            for item in sets
        ]
        primary = ordered_sets[0] if ordered_sets else ()
        sessions = [self.ledger.event_by_id[event_id].session_id for event_id in primary]
        session_indices = [self.ledger.session_order[session_id] for session_id in sessions]
        gap = max(session_indices) - min(session_indices) if session_indices else 0
        supporting = tuple(
            sorted(
                {
                    event_id
                    for event_id in qa.supporting_event_ids
                    if self.ledger.event_is_visible(event_id, qa.memory_cutoff)
                },
                key=self.ledger.event_order.__getitem__,
            )
        )
        negative_events = tuple(
            sorted(
                {
                    item
                    for item in qa.hard_negatives
                    if item in self.ledger.event_by_id and self.ledger.event_is_visible(item, qa.memory_cutoff)
                },
                key=self.ledger.event_order.__getitem__,
            )
        )
        known_images = {image.image_id for image in self.ledger.episode.images}
        negative_images = tuple(
            sorted(
                {
                    (
                        self.ledger.visual_fact_to_image[item]
                        if item in self.ledger.visual_fact_to_image
                        else item
                    )
                    for item in qa.hard_negatives
                    if item in known_images or item in self.ledger.visual_fact_to_image
                }
            )
        )
        negatives = tuple([*negative_events, *negative_images])
        return EvidenceResult(
            required_evidence_sets=tuple(ordered_sets),
            required_visual_fact_ids=qa.required_visual_fact_ids,
            supporting_event_ids=supporting,
            hard_negatives=negatives,
            hard_negative_event_ids=negative_events,
            hard_negative_image_ids=negative_images,
            clue_count=len(primary),
            distinct_evidence_sessions=len(set(sessions)),
            session_gap=gap,
            requires_long_term_memory=gap > 0,
        )
