"""Incremental, provenance-bearing observed fact ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .models import Episode, MemoryCutoff, ObservedFact


@dataclass(frozen=True)
class LedgerView:
    cutoff: MemoryCutoff
    facts: tuple[ObservedFact, ...]
    active_fact_ids: frozenset[str]

    def active_facts(self) -> list[ObservedFact]:
        return [fact for fact in self.facts if fact.fact_id in self.active_fact_ids]


class ObservedFactLedger:
    """Fact ledger whose contents can only originate from observed provenance."""

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.event_by_id = {event.event_id: event for event in episode.events}
        self.event_order = {event.event_id: index for index, event in enumerate(episode.events)}
        self.session_order = {session.session_id: index for index, session in enumerate(episode.sessions)}
        self.visual_fact_to_image: dict[str, str] = {}
        for image in episode.images:
            for visual_fact in image.private_verified_visual_facts:
                if visual_fact.visual_fact_id in self.visual_fact_to_image:
                    raise ValueError(f"duplicate visual_fact_id {visual_fact.visual_fact_id!r}")
                self.visual_fact_to_image[visual_fact.visual_fact_id] = image.image_id
        self.facts: dict[str, ObservedFact] = {}
        for fact in episode.observed_facts:
            self.add(fact)

    def add(self, fact: ObservedFact) -> None:
        if fact.fact_id in self.facts:
            raise ValueError(f"duplicate fact_id {fact.fact_id!r}")
        if fact.valid_from_session not in self.session_order:
            raise ValueError(f"fact {fact.fact_id!r} has unknown valid_from_session {fact.valid_from_session!r}")
        if fact.valid_to_session and fact.valid_to_session not in self.session_order:
            raise ValueError(f"fact {fact.fact_id!r} has unknown valid_to_session {fact.valid_to_session!r}")
        for provenance in fact.observed_provenance:
            event = self.event_by_id.get(provenance.event_id)
            if event is None:
                raise ValueError(f"fact {fact.fact_id!r} references unknown event {provenance.event_id!r}")
            for span in provenance.text_spans:
                if span not in event.content:
                    raise ValueError(
                        f"fact {fact.fact_id!r} provenance span is not verbatim in {provenance.event_id!r}: {span!r}"
                    )
            for visual_fact_id in provenance.visual_fact_ids:
                if visual_fact_id not in self.visual_fact_to_image:
                    raise ValueError(f"fact {fact.fact_id!r} references unknown visual fact {visual_fact_id!r}")
                image_id = self.visual_fact_to_image[visual_fact_id]
                if image_id not in event.image_ids:
                    raise ValueError(
                        f"fact {fact.fact_id!r} uses visual fact {visual_fact_id!r} from an image not attached to "
                        f"event {provenance.event_id!r}"
                    )
        for related in (*fact.supersedes, *fact.contradicts):
            if related == fact.fact_id:
                raise ValueError(f"fact {fact.fact_id!r} cannot relate to itself")
            if related not in self.facts:
                raise ValueError(
                    f"fact {fact.fact_id!r} references {related!r} before it exists; facts must be incremental"
                )
        self.facts[fact.fact_id] = fact

    def cutoff_event_index(self, cutoff: MemoryCutoff) -> int:
        if not self.episode.events:
            return -1
        if cutoff.mode == "episode_end":
            return len(self.episode.events) - 1
        if cutoff.mode == "event":
            if cutoff.event_id not in self.event_order:
                raise ValueError(f"unknown cutoff event_id {cutoff.event_id!r}")
            return self.event_order[str(cutoff.event_id)]
        if cutoff.mode == "session_end":
            if cutoff.session_id not in self.session_order:
                raise ValueError(f"unknown cutoff session_id {cutoff.session_id!r}")
            return max(
                (
                    index
                    for event_id, index in self.event_order.items()
                    if self.event_by_id[event_id].session_id == cutoff.session_id
                ),
                default=-1,
            )
        if cutoff.mode == "timestamp":
            if not cutoff.timestamp:
                raise ValueError("timestamp cutoff requires timestamp")
            return max(
                (
                    index
                    for event_id, index in self.event_order.items()
                    if self.event_by_id[event_id].timestamp <= cutoff.timestamp
                ),
                default=-1,
            )
        raise ValueError(f"unsupported cutoff mode {cutoff.mode!r}")

    def event_is_visible(self, event_id: str, cutoff: MemoryCutoff) -> bool:
        if event_id not in self.event_order:
            return False
        return self.event_order[event_id] <= self.cutoff_event_index(cutoff)

    def _session_visible(self, session_id: str, cutoff_index: int) -> bool:
        return any(
            event.session_id == session_id and self.event_order[event.event_id] <= cutoff_index
            for event in self.episode.events
        )

    def view(self, cutoff: MemoryCutoff) -> LedgerView:
        cutoff_index = self.cutoff_event_index(cutoff)
        visible: list[ObservedFact] = []
        for fact in self.facts.values():
            if not self._session_visible(fact.valid_from_session, cutoff_index):
                continue
            if fact.valid_to_session and self._session_visible(fact.valid_to_session, cutoff_index):
                cutoff_session = (
                    self.event_by_id[self.episode.events[cutoff_index].event_id].session_id
                    if cutoff_index >= 0
                    else ""
                )
                if self.session_order.get(cutoff_session, -1) > self.session_order[fact.valid_to_session]:
                    continue
            if not any(self.event_order[p.event_id] <= cutoff_index for p in fact.observed_provenance):
                continue
            visible.append(fact)
        visible_ids = {fact.fact_id for fact in visible}
        superseded = {
            old_id
            for fact in visible
            for old_id in fact.supersedes
            if old_id in visible_ids
        }
        active = {
            fact.fact_id
            for fact in visible
            if fact.fact_id not in superseded and fact.lifecycle_status != "retracted"
        }
        return LedgerView(cutoff=cutoff, facts=tuple(visible), active_fact_ids=frozenset(active))

    def facts_matching(
        self,
        cutoff: MemoryCutoff,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        active_only: bool = True,
    ) -> list[ObservedFact]:
        view = self.view(cutoff)
        facts: Iterable[ObservedFact] = view.active_facts() if active_only else view.facts
        result = [
            fact
            for fact in facts
            if (subject is None or fact.subject == subject) and (predicate is None or fact.predicate == predicate)
        ]
        return sorted(
            result,
            key=lambda fact: max(self.event_order[p.event_id] for p in fact.observed_provenance),
        )

    def latest(self, cutoff: MemoryCutoff, *, subject: str, predicate: str) -> ObservedFact | None:
        matches = self.facts_matching(cutoff, subject=subject, predicate=predicate, active_only=True)
        return matches[-1] if matches else None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "fact_count": len(self.facts),
            "fact_ids": list(self.facts),
            "event_count": len(self.event_by_id),
            "visual_fact_count": len(self.visual_fact_to_image),
        }
