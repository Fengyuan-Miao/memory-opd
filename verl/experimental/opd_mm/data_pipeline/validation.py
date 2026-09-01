"""Hard validators and quality certificates for accepted MMem QAs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .evidence import EvidenceResult
from .ledger import ObservedFactLedger
from .models import Episode, QACandidate
from .oracles import OracleResult


TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
GENERIC_VISUAL_VALUES = {
    "yes", "no", "true", "false", "person", "people", "man", "woman", "child",
    "dog", "cat", "image", "photo", "indoor", "outdoor", "red", "blue", "green",
    "black", "white",
}


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(value) if len(token) > 1]


def _longest_common_token_span(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    longest = 0
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            size = previous[index - 1] + 1 if left_token == right_token else 0
            current.append(size)
            longest = max(longest, size)
        previous = current
    return longest


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


def validate_episode(episode: Episode, ledger: ObservedFactLedger) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    session_ids = [session.session_id for session in episode.sessions]
    event_ids = [event.event_id for event in episode.events]
    image_ids = [image.image_id for image in episode.images]
    qa_ids = [qa.qa_id for qa in episode.qa_candidates]
    for values, label in ((session_ids, "session"), (event_ids, "event"), (image_ids, "image"), (qa_ids, "qa")):
        duplicates = sorted({item for item in values if values.count(item) > 1})
        if duplicates:
            issues.append(ValidationIssue(f"duplicate_{label}_id", f"duplicate {label} IDs: {duplicates}"))
    for session in episode.sessions:
        turns = [event.turn_index for event in session.events]
        if turns != sorted(turns) or len(turns) != len(set(turns)):
            issues.append(
                ValidationIssue(
                    "invalid_turn_order",
                    f"{session.session_id} turn_index values are not unique/increasing",
                )
            )
    known_images = set(image_ids)
    for event in episode.events:
        missing = sorted(set(event.image_ids) - known_images)
        if missing:
            issues.append(
                ValidationIssue(
                    "unknown_event_image",
                    f"{event.event_id} references unknown images: {missing}",
                )
            )
        for image_id in event.image_ids:
            image = next(item for item in episode.images if item.image_id == image_id)
            if image.role == "query":
                issues.append(
                    ValidationIssue(
                        "query_image_in_memory",
                        f"query-only image {image_id} is attached to {event.event_id}",
                    )
                )
    for image in episode.images:
        public = " ".join(_tokens(image.public_retrieval_description))
        for fact in image.private_verified_visual_facts:
            value = " ".join(_tokens(str(fact.value)))
            if len(value) >= 4 and value not in GENERIC_VISUAL_VALUES and value in public:
                issues.append(
                    ValidationIssue(
                        "private_visual_fact_leak",
                        f"image {image.image_id} public description contains private visual value "
                        f"for {fact.visual_fact_id}",
                    )
                )
    if not ledger.facts:
        issues.append(ValidationIssue("empty_fact_ledger", "episode has no observed facts"))
    return issues


def _evidence_covers_facts(
    event_set: Iterable[str],
    required_fact_ids: Iterable[str],
    ledger: ObservedFactLedger,
) -> bool:
    events = set(event_set)
    for fact_id in required_fact_ids:
        fact = ledger.facts.get(fact_id)
        if fact is None or not any(provenance.event_id in events for provenance in fact.observed_provenance):
            return False
    return True


def _evidence_covers_visual_facts(
    event_set: Iterable[str],
    required_visual_fact_ids: Iterable[str],
    ledger: ObservedFactLedger,
    query_image_ids: Iterable[str],
) -> bool:
    events = set(event_set)
    query_images = set(query_image_ids)
    for visual_fact_id in required_visual_fact_ids:
        image_id = ledger.visual_fact_to_image.get(visual_fact_id)
        if image_id is None:
            return False
        if image_id in query_images:
            continue
        if not any(image_id in ledger.event_by_id[event_id].image_ids for event_id in events):
            return False
    return True


def validate_qa(
    qa: QACandidate,
    *,
    episode: Episode,
    ledger: ObservedFactLedger,
    oracle: OracleResult,
    evidence: EvidenceResult,
    max_query_support_overlap: float = 0.85,
    max_common_span: int = 12,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    event_ids = set(ledger.event_by_id)
    image_ids = {image.image_id for image in episode.images}
    if not oracle.valid:
        issues.extend(ValidationIssue("oracle_failure", message) for message in oracle.errors)
    unknown_query_images = sorted(set(qa.question_image_ids) - image_ids)
    if unknown_query_images:
        issues.append(ValidationIssue("unknown_query_image", f"unknown query images: {unknown_query_images}"))
    memory_images = {image_id for event in episode.events for image_id in event.image_ids}
    overlap = sorted(set(qa.question_image_ids) & memory_images)
    if overlap:
        issues.append(
            ValidationIssue(
                "query_image_is_memory",
                f"query image IDs also occur in memory events: {overlap}",
            )
        )
    if qa.question_image_ids and qa.question_modality not in {"image", "text+image"}:
        issues.append(
            ValidationIssue(
                "question_modality_mismatch",
                "question_image_ids are present but modality is text-only",
            )
        )
    if not qa.question_image_ids and qa.question_modality in {"image", "text+image"}:
        issues.append(ValidationIssue("question_modality_mismatch", "image modality is declared without a query image"))

    for event_set in evidence.required_evidence_sets:
        unknown = sorted(set(event_set) - event_ids)
        if unknown:
            issues.append(ValidationIssue("unknown_evidence_event", f"unknown evidence events: {unknown}"))
            continue
        future = [event_id for event_id in event_set if not ledger.event_is_visible(event_id, qa.memory_cutoff)]
        if future:
            issues.append(ValidationIssue("post_cutoff_evidence", f"evidence is after memory cutoff: {future}"))
        if oracle.required_fact_ids and not _evidence_covers_facts(event_set, oracle.required_fact_ids, ledger):
            issues.append(
                ValidationIssue(
                    "insufficient_evidence",
                    f"evidence set does not cover oracle facts: {event_set}",
                )
            )
        if qa.required_visual_fact_ids and not _evidence_covers_visual_facts(
            event_set,
            qa.required_visual_fact_ids,
            ledger,
            qa.question_image_ids,
        ):
            issues.append(
                ValidationIssue(
                    "insufficient_visual_evidence",
                    f"evidence set does not cover required visual facts: {event_set}",
                )
            )
        if (oracle.required_fact_ids or qa.required_visual_fact_ids) and len(event_set) > 1:
            for event_id in event_set:
                reduced = [item for item in event_set if item != event_id]
                facts_still_covered = _evidence_covers_facts(reduced, oracle.required_fact_ids, ledger)
                visuals_still_covered = _evidence_covers_visual_facts(
                    reduced,
                    qa.required_visual_fact_ids,
                    ledger,
                    qa.question_image_ids,
                )
                if facts_still_covered and visuals_still_covered:
                    issues.append(
                        ValidationIssue(
                            "nonminimal_evidence",
                            f"{event_id} can be removed from evidence set {event_set}",
                        )
                    )
                    break
    if qa.task != "AR" and not evidence.required_evidence_sets:
        issues.append(ValidationIssue("empty_positive_evidence", "positive factual QA has no required evidence set"))
    if set(evidence.hard_negatives) & {item for group in evidence.required_evidence_sets for item in group}:
        issues.append(ValidationIssue("negative_is_evidence", "a hard negative is also required evidence"))

    support_text = " ".join(
        ledger.event_by_id[event_id].content
        for event_id in (evidence.required_evidence_sets[0] if evidence.required_evidence_sets else ())
    )
    query_tokens = _tokens(qa.question_text)
    support_tokens = _tokens(support_text)
    if query_tokens and support_tokens:
        query_set = set(query_tokens)
        overlap_ratio = len(query_set & set(support_tokens)) / len(query_set)
        common_span = _longest_common_token_span(query_tokens, support_tokens)
        if overlap_ratio >= max_query_support_overlap and common_span >= max_common_span:
            issues.append(
                ValidationIssue(
                    "query_support_restatement",
                    f"question copies support too closely "
                    f"(token_overlap={overlap_ratio:.3f}, common_span={common_span})",
                )
            )
    answer_tokens = _tokens(qa.answer)
    answer_phrase = " ".join(answer_tokens)
    question_phrase = " ".join(query_tokens)
    if answer_phrase and answer_phrase in question_phrase and qa.task != "CD":
        issues.append(ValidationIssue("answer_leak_in_query", "answer appears verbatim in the question"))
    if qa.task == "AR" and qa.required_visual_fact_ids:
        issues.append(
            ValidationIssue(
                "invalid_ar_visual_dependency",
                "AR must be certified by the closed-world ledger, not a hidden visual fact",
            )
        )
    if qa.task == "AR" and re.match(
        r"^\s*(?:based\b.*?[,,:]\s*)?(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b",
        qa.question_text,
        flags=re.IGNORECASE,
    ):
        issues.append(
            ValidationIssue(
                "ar_binary_claim_question",
                "AR must request a missing attribute value rather than ask whether a claim is true",
            )
        )
    if qa.required_visual_fact_ids:
        dialogue_text = "\n".join(event.content for event in episode.events).casefold()
        visual_values = {
            fact.visual_fact_id: fact.value
            for image in episode.images
            for fact in image.private_verified_visual_facts
        }
        for visual_id in qa.required_visual_fact_ids:
            value = str(visual_values.get(visual_id, "")).strip().casefold()
            if value and value not in {"true", "false", "yes", "no"}:
                if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", dialogue_text):
                    issues.append(
                        ValidationIssue(
                            "visual_target_in_dialogue",
                            f"visual target {visual_id} is directly stated in dialogue",
                        )
                    )
                    break
    if qa.task == "VS" and len(episode.images) > 1 and not evidence.hard_negative_image_ids:
        issues.append(
            ValidationIssue(
                "missing_visual_hard_negative",
                "VS requires at least one different historical-image hard negative",
            )
        )
    return issues


def quality_certificate(
    qa: QACandidate,
    *,
    oracle: OracleResult,
    evidence: EvidenceResult,
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    errors = [issue for issue in issues if issue.severity == "error"]
    structural_ablation = {
        "full_evidence_oracle_valid": oracle.valid,
        "remove_one_event_breaks_required_fact_coverage": not any(
            issue.code == "nonminimal_evidence" for issue in issues
        ),
        "closed_book": "diagnostic_not_run",
        "local_window": "diagnostic_not_run",
        "text_only": "not_applicable" if not qa.required_visual_fact_ids else "diagnostic_not_run",
        "image_only": "not_applicable" if not qa.required_visual_fact_ids else "diagnostic_not_run",
    }
    return {
        "qa_id": qa.qa_id,
        "accepted": not errors,
        "task": qa.task,
        "memory_cutoff": qa.memory_cutoff.to_dict(),
        "oracle": oracle.to_dict(),
        "evidence": evidence.to_dict(),
        "ablation": structural_ablation,
        "checks": {
            "schema_valid": True,
            "oracle_valid": oracle.valid,
            "cutoff_valid": not any(issue.code == "post_cutoff_evidence" for issue in errors),
            "minimal_evidence": not any(issue.code == "nonminimal_evidence" for issue in errors),
            "query_not_restatement": not any(issue.code == "query_support_restatement" for issue in errors),
            "no_answer_leak": not any(issue.code == "answer_leak_in_query" for issue in errors),
            "visual_target_not_in_dialogue": not any(
                issue.code == "visual_target_in_dialogue" for issue in errors
            ),
            "visual_hard_negative_present": not any(
                issue.code == "missing_visual_hard_negative" for issue in errors
            ),
        },
        "issues": [issue.to_dict() for issue in issues],
    }
