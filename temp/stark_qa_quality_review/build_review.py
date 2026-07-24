"""Attach manual quality-review labels to the fixed STARK QA sample."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "sample_90_raw.jsonl"
REVIEWED_PATH = ROOT / "sample_90_reviewed.jsonl"
SUMMARY_PATH = ROOT / "review_summary.json"


# Labels not listed here are clean passes. The review examined the supplied
# support memories for every item, the full episode for AR items, and the
# actual image files for VS, TTL, and VR items.
REVIEW_OVERRIDES = {
    # FR: the answer may exist elsewhere in the episode, but the declared gold
    # support does not establish the fact asked by the question.
    "stark:158_281d3e83-7207-4147-81fc-eecb3f42e1f8-characteristic:qa:0000": (
        "fail",
        ["support_incomplete"],
        "The support says Lake Tegernsee is beautiful, but never calls it a favorite hideaway.",
    ),
    "stark:188_d11c9a78-77fd-4541-aa11-047f85e4b4ac-experience:qa:0000": (
        "fail",
        ["support_incomplete", "unresolved_reference"],
        "The support only says 'that book' and never names The Picture of Dorian Gray.",
    ),
    "stark:45_26f17633-8c45-4044-842d-d6e461c88e97-goal:qa:0000": (
        "fail",
        ["support_incomplete", "question_conflates_facts"],
        "The support establishes teaching Italian dishes, but not the cookbook or preservation goal.",
    ),
    # VS: answers identify the intended images, but wording or event-specific
    # semantics are weaker than the visible content.
    "stark:129_0af8ef73-a928-4102-beb2-b89f9e3f0955-characteristic:qa:0001": (
        "minor_issue",
        ["metadata_wording"],
        "Correct image ID; 'available image' is unnecessary system-like wording.",
    ),
    "stark:152_c6cf24bd-9d87-4c7e-9c3f-db3b407b5035-experience:qa:0001": (
        "minor_issue",
        ["metadata_wording"],
        "Correct image ID; 'available image' is unnecessary system-like wording.",
    ),
    "stark:178_7ac60707-e40c-4646-b9af-22a15b12b634-relationship:qa:0001": (
        "minor_issue",
        ["metadata_wording", "weak_visual_specificity"],
        "The images show friends and a cat, but trivia/game-night context is not visually explicit.",
    ),
    "stark:57_5d539c57-d8f9-400c-a632-c65af30e7024-experience:qa:0001": (
        "minor_issue",
        ["weak_visual_specificity"],
        "The award is visible, while the second image does not unambiguously depict a scholarship launch.",
    ),
    # TTL: these concepts cannot all be inferred reliably from pixels alone.
    "stark:73_752bbddb-9013-464e-83cd-4fa1b8383b03-routine:qa:0002": (
        "minor_issue",
        ["weak_visual_grounding"],
        "The image appears intended as acupuncture, but the generated metal objects are visually ambiguous.",
    ),
    "stark:73_784588f7-adf8-429e-9e58-ecf52b6964e1-goal:qa:0002": (
        "fail",
        ["image_not_diagnostic"],
        "A generic speaker image does not visually establish a diversity-and-inclusion advocacy theme.",
    ),
    "stark:94_a5084d3d-4b84-4188-bd39-a3cb505d0021-goal:qa:0002": (
        "fail",
        ["image_not_diagnostic"],
        "Two people talking in a cafe does not visually establish that this is a Russian language exchange.",
    ),
    # VR: the count matches the captions, but identity across ages is not
    # independently verifiable from the images.
    "stark:73_7bcc5b56-eb1f-4075-8c45-2927a3a9893f-goal:qa:0003": (
        "minor_issue",
        ["identity_not_visually_verifiable"],
        "All five are intended to depict Guan, but cross-age identity cannot be verified from pixels alone.",
    ),
    # MR: declared support names Venice but not the grandchildren.
    "stark:178_7de6f053-24d7-4131-85e9-d98c26b1871d-goal:qa:0005": (
        "fail",
        ["support_incomplete"],
        "The support establishes Venice and documenting memories, but never says grandchildren traveled with her.",
    ),
    # CD: answers are correct, but these are direct fact checks rather than
    # consistently phrased contradiction judgments.
    "stark:173_76767bf4-ab3b-4dae-804d-11a620e489df-goal:qa:0006": (
        "minor_issue",
        ["category_style_drift"],
        "Factually correct, but phrased as a direct yes/no recall question rather than conflict detection.",
    ),
    "stark:178_7de6f053-24d7-4131-85e9-d98c26b1871d-goal:qa:0007": (
        "minor_issue",
        ["category_style_drift"],
        "Factually correct, but phrased as a direct yes/no recall question rather than conflict detection.",
    ),
}


def main() -> None:
    rows = [json.loads(line) for line in RAW_PATH.read_text().splitlines() if line.strip()]
    reviewed = []
    for row in rows:
        label, issues, notes = REVIEW_OVERRIDES.get(
            row["sample_id"],
            ("pass", [], "Question and concise answer are supported by the declared evidence."),
        )
        item = dict(row)
        item["review"] = {
            "label": label,
            "issues": issues,
            "notes": notes,
            "scope": (
                "support memories + full episode absence check"
                if row["point"] == "AR"
                else "support memories + actual image files"
                if row["point"] in {"VS", "TTL", "VR"}
                else "support memories"
            ),
        }
        reviewed.append(item)

    REVIEWED_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed)
    )

    by_point: dict[str, Counter] = defaultdict(Counter)
    labels = Counter()
    issues = Counter()
    for row in reviewed:
        point = row["point"]
        label = row["review"]["label"]
        by_point[point][label] += 1
        labels[label] += 1
        issues.update(row["review"]["issues"])

    summary = {
        "sample_file": str(RAW_PATH),
        "reviewed_file": str(REVIEWED_PATH),
        "review_method": (
            "Manual review of declared support for all 90 QAs; actual images for VS/TTL/VR; "
            "full-episode absence checks for AR."
        ),
        "sample_count": len(reviewed),
        "label_counts": dict(labels),
        "label_rates": {key: value / len(reviewed) for key, value in labels.items()},
        "per_point": {
            point: {
                "pass": counts["pass"],
                "minor_issue": counts["minor_issue"],
                "fail": counts["fail"],
            }
            for point, counts in sorted(by_point.items())
        },
        "issue_counts": dict(issues.most_common()),
        "conclusion": (
            "Most sampled QAs are usable. The main blocking defects are incomplete declared support "
            "and visually non-diagnostic TTL questions; VS wording and CD style drift are lower-severity issues."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
