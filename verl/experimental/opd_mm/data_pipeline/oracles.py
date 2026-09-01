"""Programmatic task oracles used before a generated QA can be accepted."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .ledger import ObservedFactLedger
from .models import ObservedFact, QACandidate


def normalize_answer(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip().casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


@dataclass(frozen=True)
class OracleResult:
    valid: bool
    computed_answer: str
    required_fact_ids: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "computed_answer": self.computed_answer,
            "required_fact_ids": list(self.required_fact_ids),
            "errors": list(self.errors),
        }


def _format_objects(facts: list[ObservedFact], oracle: dict[str, Any]) -> str:
    values = [str(fact.object) for fact in facts]
    if oracle.get("sort") is True:
        values.sort(key=str.casefold)
    return str(oracle.get("answer_joiner", ", ")).join(values)


def _fact_lookup(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    required = oracle.get("required_fact_ids")
    if required is None and oracle.get("fact_id"):
        required = [oracle["fact_id"]]
    if isinstance(required, list) and required:
        view = ledger.view(qa.memory_cutoff)
        visible = {fact.fact_id: fact for fact in view.facts}
        missing = [str(fact_id) for fact_id in required if str(fact_id) not in visible]
        if missing:
            return OracleResult(False, "", tuple(str(item) for item in required), (f"facts not visible: {missing}",))
        facts = [visible[str(fact_id)] for fact_id in required]
    else:
        subject = oracle.get("subject")
        predicate = oracle.get("predicate")
        if not subject or not predicate:
            return OracleResult(False, "", (), ("fact_lookup requires fact_id(s) or subject+predicate",))
        facts = ledger.facts_matching(
            qa.memory_cutoff,
            subject=str(subject),
            predicate=str(predicate),
            active_only=bool(oracle.get("active_only", True)),
        )
        if not facts:
            return OracleResult(False, "", (), ("no fact matches subject+predicate at cutoff",))
        if oracle.get("select", "latest") == "latest":
            facts = facts[-1:]
    answer = _format_objects(facts, oracle)
    return OracleResult(
        normalize_answer(answer) == normalize_answer(qa.answer),
        answer,
        tuple(fact.fact_id for fact in facts),
        () if normalize_answer(answer) == normalize_answer(qa.answer) else ("gold answer disagrees with fact oracle",),
    )


def _temporal(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    kind = str(oracle.get("kind"))
    if kind == "temporal_relation":
        left = str(oracle.get("left_event_id") or "")
        right = str(oracle.get("right_event_id") or "")
        relation = str(oracle.get("relation") or "before")
        if left not in ledger.event_order or right not in ledger.event_order:
            return OracleResult(False, "", (), ("temporal_relation references unknown event",))
        if not ledger.event_is_visible(left, qa.memory_cutoff) or not ledger.event_is_visible(right, qa.memory_cutoff):
            return OracleResult(False, "", (), ("temporal_relation references post-cutoff event",))
        before = ledger.event_order[left] < ledger.event_order[right]
        truth = before if relation == "before" else not before if relation == "after" else None
        if truth is None:
            return OracleResult(False, "", (), ("relation must be before or after",))
        answer = str(oracle.get("true_answer", "Yes") if truth else oracle.get("false_answer", "No"))
        valid = normalize_answer(answer) == normalize_answer(qa.answer)
        return OracleResult(valid, answer, (), () if valid else ("gold answer disagrees with temporal relation",))
    if kind == "temporal_order":
        event_ids = oracle.get("event_ids")
        if not isinstance(event_ids, list) or len(event_ids) < 2:
            return OracleResult(False, "", (), ("temporal_order requires at least two event_ids",))
        event_ids = [str(item) for item in event_ids]
        if any(item not in ledger.event_order for item in event_ids):
            return OracleResult(False, "", (), ("temporal_order references unknown event",))
        if any(not ledger.event_is_visible(item, qa.memory_cutoff) for item in event_ids):
            return OracleResult(False, "", (), ("temporal_order references post-cutoff event",))
        labels = oracle.get("labels")
        label_by_event = (
            {event_id: str(label) for event_id, label in zip(event_ids, labels, strict=True)}
            if isinstance(labels, list) and len(labels) == len(event_ids)
            else {event_id: event_id for event_id in event_ids}
        )
        ordered = sorted(event_ids, key=ledger.event_order.__getitem__, reverse=bool(oracle.get("descending", False)))
        answer = str(oracle.get("answer_joiner", ", ")).join(label_by_event[event_id] for event_id in ordered)
        valid = normalize_answer(answer) == normalize_answer(qa.answer)
        return OracleResult(valid, answer, (), () if valid else ("gold answer disagrees with temporal order",))
    return OracleResult(False, "", (), (f"unsupported temporal oracle {kind!r}",))


def _latest_valid_value(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    subject = str(oracle.get("subject") or "")
    predicate = str(oracle.get("predicate") or "")
    if not subject or not predicate:
        return OracleResult(False, "", (), ("latest_valid_value requires subject and predicate",))
    latest = ledger.latest(qa.memory_cutoff, subject=subject, predicate=predicate)
    if latest is None:
        return OracleResult(False, "", (), ("no active value exists at cutoff",))
    new_fact_id = oracle.get("new_fact_id")
    errors = []
    if new_fact_id and latest.fact_id != str(new_fact_id):
        errors.append(f"latest fact is {latest.fact_id!r}, not declared new_fact_id")
    old_fact_id = oracle.get("old_fact_id")
    if old_fact_id and str(old_fact_id) not in latest.supersedes:
        errors.append("new fact does not explicitly supersede old_fact_id")
    answer = str(latest.object)
    if normalize_answer(answer) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with latest valid value")
    required = tuple(str(item) for item in (old_fact_id, latest.fact_id) if item)
    return OracleResult(not errors, answer, required, tuple(errors))


def _absence(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    scope = oracle.get("closed_world_scope")
    predicate = str(oracle.get("missing_predicate") or "")
    if not isinstance(scope, list) or not scope or predicate not in {str(item) for item in scope}:
        return OracleResult(False, "", (), ("absence oracle requires missing_predicate inside closed_world_scope",))
    anchors = oracle.get("topic_anchor_event_ids")
    if not isinstance(anchors, list) or not anchors:
        return OracleResult(False, "", (), ("absence oracle requires topic_anchor_event_ids",))
    if any(str(event_id) not in ledger.event_order for event_id in anchors):
        return OracleResult(False, "", (), ("absence oracle references unknown anchor event",))
    if any(not ledger.event_is_visible(str(event_id), qa.memory_cutoff) for event_id in anchors):
        return OracleResult(False, "", (), ("absence oracle references post-cutoff anchor",))
    subject = str(oracle["subject"]) if oracle.get("subject") else None
    matches = ledger.facts_matching(
        qa.memory_cutoff,
        subject=subject,
        predicate=predicate,
        active_only=False,
    )
    expected = str(oracle.get("absence_answer") or "Not mentioned.")
    errors = []
    if matches:
        errors.append(f"closed-world audit found {len(matches)} matching fact(s)")
    if normalize_answer(expected) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with absence oracle")
    return OracleResult(not errors, expected, (), tuple(errors))


def _image_and_visual_fact(ledger: ObservedFactLedger, visual_fact_id: str):
    image_id = ledger.visual_fact_to_image.get(visual_fact_id)
    if image_id is None:
        return None, None
    image = next((item for item in ledger.episode.images if item.image_id == image_id), None)
    if image is None:
        return None, None
    visual = next(
        (item for item in image.private_verified_visual_facts if item.visual_fact_id == visual_fact_id),
        None,
    )
    return image, visual


def _visual_is_available(ledger: ObservedFactLedger, qa: QACandidate, image_id: str) -> bool:
    image = next((item for item in ledger.episode.images if item.image_id == image_id), None)
    if image is None:
        return False
    if image.role in {"query", "both"} and image_id in qa.question_image_ids:
        return True
    return any(
        image_id in event.image_ids and ledger.event_is_visible(event.event_id, qa.memory_cutoff)
        for event in ledger.episode.events
    )


def _image_lookup(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    image_id = str(oracle.get("image_id") or "")
    image = next((item for item in ledger.episode.images if item.image_id == image_id), None)
    errors = []
    if image is None or image.role == "query":
        errors.append("image_lookup requires an existing memory image")
    elif not _visual_is_available(ledger, qa, image_id):
        errors.append("image_lookup target is not visible at cutoff")
    required_visual = [str(item) for item in oracle.get("required_visual_fact_ids", [])]
    image_visual_ids = {
        item.visual_fact_id for item in image.private_verified_visual_facts
    } if image is not None else set()
    if not required_visual or not set(required_visual) <= image_visual_ids:
        errors.append("image_lookup visual facts are missing or belong to another image")
    if set(required_visual) != set(qa.required_visual_fact_ids):
        errors.append("QA required_visual_fact_ids disagree with image_lookup oracle")
    if normalize_answer(image_id) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with image_lookup")
    query_image_id = str(oracle.get("query_image_id") or "")
    if query_image_id:
        query_image = next((item for item in ledger.episode.images if item.image_id == query_image_id), None)
        if (
            query_image is None
            or query_image.role != "query"
            or query_image_id not in qa.question_image_ids
        ):
            errors.append("image_lookup query_image_id must be an attached query-only image")
        predicates = [str(item) for item in oracle.get("matching_visual_predicates", [])]
        if not predicates:
            errors.append("query-image lookup requires matching_visual_predicates")
        elif image is not None and query_image is not None:
            for predicate in predicates:
                target_values = {
                    normalize_answer(item.value)
                    for item in image.private_verified_visual_facts
                    if item.predicate == predicate
                }
                query_values = {
                    normalize_answer(item.value)
                    for item in query_image.private_verified_visual_facts
                    if item.predicate == predicate
                }
                if not target_values or not query_values or target_values.isdisjoint(query_values):
                    errors.append(f"query and target images do not share predicate {predicate!r}")
    required_facts = tuple(str(item) for item in oracle.get("required_fact_ids", []))
    return OracleResult(not errors, image_id, required_facts, tuple(errors))


def _visual_rule_match(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    image_id = str(oracle.get("query_image_id") or "")
    predicate = str(oracle.get("predicate") or "")
    image = next((item for item in ledger.episode.images if item.image_id == image_id), None)
    errors = []
    if image is None or image.role not in {"query", "both"} or image_id not in qa.question_image_ids:
        errors.append("visual_rule_match requires the declared query image")
        matches = False
    else:
        values = [item.value for item in image.private_verified_visual_facts if item.predicate == predicate]
        if not values:
            errors.append("query image lacks the verified predicate required by the visual rule")
        expected = normalize_answer(oracle.get("expected_value"))
        matches = any(normalize_answer(value) == expected for value in values)
    answer = str(oracle.get("true_answer", "Yes") if matches else oracle.get("false_answer", "No"))
    if normalize_answer(answer) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with visual_rule_match")
    required = tuple(str(item) for item in oracle.get("required_fact_ids", []))
    return OracleResult(not errors, answer, required, tuple(errors))


def _visual_text_relation(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    fact_id = str(oracle.get("fact_id") or "")
    visual_fact_id = str(oracle.get("visual_fact_id") or "")
    view = ledger.view(qa.memory_cutoff)
    fact = next((item for item in view.facts if item.fact_id == fact_id), None)
    image, visual = _image_and_visual_fact(ledger, visual_fact_id)
    errors = []
    if fact is None:
        errors.append("visual_text_relation references a fact not visible at cutoff")
    if image is None or visual is None or not _visual_is_available(ledger, qa, image.image_id):
        errors.append("visual_text_relation references an unavailable visual fact")
    equal = fact is not None and visual is not None and normalize_answer(fact.object) == normalize_answer(visual.value)
    relation = str(oracle.get("relation") or "equals")
    if relation not in {"equals", "not_equals"}:
        errors.append("visual_text_relation relation must be equals or not_equals")
    truth = equal if relation == "equals" else not equal
    answer = str(oracle.get("true_answer", "Yes") if truth else oracle.get("false_answer", "No"))
    if normalize_answer(answer) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with visual_text_relation")
    if visual_fact_id not in qa.required_visual_fact_ids:
        errors.append("visual relation dependency is missing from required_visual_fact_ids")
    return OracleResult(not errors, answer, (fact_id,) if fact_id else (), tuple(errors))


def _conflict_detection(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    oracle = qa.task_oracle
    left_id = str(oracle.get("left_fact_id") or "")
    right_id = str(oracle.get("right_fact_id") or "")
    view = ledger.view(qa.memory_cutoff)
    visible = {fact.fact_id: fact for fact in view.facts}
    left, right = visible.get(left_id), visible.get(right_id)
    errors = []
    if left is None or right is None:
        errors.append("conflict_detection facts must both be visible at cutoff")
        conflict = False
    else:
        same_slot = left.subject == right.subject and left.predicate == right.predicate
        sequential_update = left_id in right.supersedes or right_id in left.supersedes
        incompatible = normalize_answer(left.object) != normalize_answer(right.object)
        conflict = same_slot and incompatible and not sequential_update
        if not same_slot:
            errors.append("conflict_detection facts do not describe the same subject/predicate")
    answer = str(oracle.get("true_answer", "Yes") if conflict else oracle.get("false_answer", "No"))
    if normalize_answer(answer) != normalize_answer(qa.answer):
        errors.append("gold answer disagrees with conflict_detection")
    return OracleResult(not errors, answer, tuple(item for item in (left_id, right_id) if item), tuple(errors))


def evaluate_task_oracle(qa: QACandidate, ledger: ObservedFactLedger) -> OracleResult:
    """Evaluate a QA against the full observed graph at its declared cutoff."""

    kind = str(qa.task_oracle.get("kind") or "")
    if qa.task == "FR" and kind in {"fact_lookup", "direct_fact"}:
        return _fact_lookup(qa, ledger)
    if qa.task == "TR" and kind in {"temporal_relation", "temporal_order"}:
        return _temporal(qa, ledger)
    if qa.task == "KR" and kind == "latest_valid_value":
        return _latest_valid_value(qa, ledger)
    if qa.task == "AR" and kind == "absence_in_closed_scope":
        return _absence(qa, ledger)
    if qa.task == "VS" and kind == "image_lookup":
        return _image_lookup(qa, ledger)
    if qa.task == "TTL" and kind == "visual_rule_match":
        return _visual_rule_match(qa, ledger)
    if qa.task == "VR" and kind == "visual_text_relation":
        return _visual_text_relation(qa, ledger)
    if qa.task == "MR" and kind == "multi_fact_lookup":
        return _fact_lookup(qa, ledger)
    if qa.task == "CD" and kind == "conflict_detection":
        return _conflict_detection(qa, ledger)
    return OracleResult(
        False,
        "",
        (),
        (f"Base-v0 has no deterministic oracle for task={qa.task!r}, kind={kind!r}",),
    )
