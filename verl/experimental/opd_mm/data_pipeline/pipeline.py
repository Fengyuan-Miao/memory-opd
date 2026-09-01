"""End-to-end deterministic acceptance pipeline for MMem Base-v0."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .evidence import EvidenceMiner
from .export import export_mem_gallery_episode, export_opd_mm_store
from .ledger import ObservedFactLedger
from .models import ArtifactValidationError, Episode
from .oracles import evaluate_task_oracle
from .validation import ValidationIssue, quality_certificate, validate_episode, validate_qa


OPAQUE_IMAGE_ID = re.compile(r"^IMG_[0-9a-f]{10}$")


@dataclass(frozen=True)
class BuildResult:
    episode: Episode
    accepted_qas: tuple[dict[str, Any], ...]
    rejected_qas: tuple[dict[str, Any], ...]
    certificates: tuple[dict[str, Any], ...]
    episode_issues: tuple[ValidationIssue, ...]

    @property
    def accepted(self) -> bool:
        return not any(issue.severity == "error" for issue in self.episode_issues) and bool(self.accepted_qas)

    def manifest(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "schema_version": self.episode.schema_version,
            "accepted": self.accepted,
            "session_count": len(self.episode.sessions),
            "event_count": len(self.episode.events),
            "image_count": len(self.episode.images),
            "observed_fact_count": len(self.episode.observed_facts),
            "qa_candidate_count": len(self.episode.qa_candidates),
            "accepted_qa_count": len(self.accepted_qas),
            "rejected_qa_count": len(self.rejected_qas),
            "episode_issues": [issue.to_dict() for issue in self.episode_issues],
        }


def build_episode(value: dict[str, Any], *, source_root: str | Path = "", strict: bool = False) -> BuildResult:
    """Validate one artifact, run task oracles, and mine event evidence."""

    episode = Episode.from_dict(value, source_root=source_root)
    ledger = ObservedFactLedger(episode)
    episode_issues = validate_episode(episode, ledger)
    miner = EvidenceMiner(ledger)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    for qa in episode.qa_candidates:
        try:
            oracle = evaluate_task_oracle(qa, ledger)
            evidence = miner.mine(qa)
            issues = [
                *episode_issues,
                *validate_qa(
                    qa,
                    episode=episode,
                    ledger=ledger,
                    oracle=oracle,
                    evidence=evidence,
                ),
            ]
            certificate = quality_certificate(qa, oracle=oracle, evidence=evidence, issues=issues)
            enriched = {**qa.to_dict(), **evidence.to_dict()}
        except Exception as exc:
            issue = ValidationIssue("qa_pipeline_error", f"{type(exc).__name__}: {exc}")
            certificate = {
                "qa_id": qa.qa_id,
                "accepted": False,
                "task": qa.task,
                "issues": [issue.to_dict()],
            }
            enriched = qa.to_dict()
        certificates.append(certificate)
        (accepted if certificate["accepted"] else rejected).append(enriched)
    result = BuildResult(
        episode=episode,
        accepted_qas=tuple(accepted),
        rejected_qas=tuple(rejected),
        certificates=tuple(certificates),
        episode_issues=tuple(episode_issues),
    )
    if strict and (episode_issues or rejected):
        messages = [issue.message for issue in episode_issues]
        messages.extend(
            f"{certificate['qa_id']}: {', '.join(item['message'] for item in certificate.get('issues', []))}"
            for certificate in certificates
            if not certificate.get("accepted")
        )
        raise ArtifactValidationError(messages)
    return result


def _load_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_public_episode(path: Path) -> list[str]:
    """Validate only model-visible boundary invariants after export."""

    value = _load_artifact(path)
    issues: list[str] = []
    profile = value.get("character_profile")
    if not isinstance(profile, dict) or set(profile) - {"name"}:
        issues.append("public character_profile must contain only the display name")
    for session in value.get("multi_session_dialogues", []):
        for turn in session.get("dialogues", []):
            if not str(turn.get("timestamp") or "").strip():
                issues.append(f"{turn.get('round')} has no public timestamp")
            if "image_caption" in turn:
                issues.append(f"{turn.get('round')} exposes an internal image caption")
            for image_id in turn.get("image_id", []):
                if OPAQUE_IMAGE_ID.fullmatch(str(image_id)) is None:
                    issues.append(f"non-opaque public image ID: {image_id}")
    provenance = value.get("annotation_provenance")
    if not isinstance(provenance, dict) or provenance.get("human_reviewed") is not False:
        issues.append("annotation provenance must explicitly record human_reviewed=false")
    for qa in value.get("human-annotated QAs", []):
        qa_id = str(qa.get("sample_id") or qa.get("question") or "QA")
        if not isinstance(qa.get("required_evidence_sets"), list):
            issues.append(f"{qa_id} has no typed required_evidence_sets")
        if not qa.get("answer_type"):
            issues.append(f"{qa_id} has no answer_type")
        if not isinstance(qa.get("memory_cutoff"), dict):
            issues.append(f"{qa_id} has no memory_cutoff")
        if qa.get("point") == "AR" and not isinstance(qa.get("evidence_scope"), dict):
            issues.append(f"{qa_id} has no AR evidence_scope")
    return issues


def build_dataset(
    artifact_paths: Iterable[str | Path],
    *,
    output_root: str | Path,
    strict: bool = False,
    prepared_artifacts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and export multiple episodes into one Mem-Gallery-compatible corpus."""

    root = Path(output_root)
    results: list[BuildResult] = []
    public_admission: list[dict[str, Any]] = []
    for path_value in artifact_paths:
        path = Path(path_value).resolve()
        value = prepared_artifacts.get(str(path)) if prepared_artifacts else None
        if value is None:
            value = _load_artifact(path)
        result = build_episode(value, source_root=path.parent, strict=strict)
        results.append(result)
        report_root = root / "reports" / result.episode.episode_id
        _write_json(report_root / "manifest.json", result.manifest())
        _write_json(report_root / "quality_certificates.json", list(result.certificates))
        _write_json(report_root / "rejected_qas.json", list(result.rejected_qas))
        if result.accepted_qas and not result.episode_issues:
            public_path = export_mem_gallery_episode(result.episode, result.accepted_qas, root)
            public_issues = _validate_public_episode(public_path)
            public_admission.append(
                {
                    "episode_id": result.episode.episode_id,
                    "accepted": not public_issues,
                    "issues": public_issues,
                }
            )
            if strict and public_issues:
                _write_json(root / "reports" / "public_admission.json", public_admission)
                raise ArtifactValidationError(public_issues)
    dataset_name = results[0].episode.dataset if results else "mmem_v2"
    if any(result.episode.dataset != dataset_name for result in results):
        raise ValueError("all episodes in one build must use the same dataset name")
    store_manifest = export_opd_mm_store(root, root / "opd_mm_store", dataset_name=dataset_name)
    _write_json(root / "reports" / "public_admission.json", public_admission)
    manifest = {
        "dataset": dataset_name,
        "schema_version": "mmem-v2.0",
        "episode_count": len(results),
        "accepted_episode_count": sum(result.accepted for result in results),
        "accepted_qa_count": sum(len(result.accepted_qas) for result in results),
        "rejected_qa_count": sum(len(result.rejected_qas) for result in results),
        "episodes": [result.manifest() for result in results],
        "opd_mm_store": store_manifest,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
