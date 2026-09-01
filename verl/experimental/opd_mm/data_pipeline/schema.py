"""Machine-readable JSON Schema for Base-v0 episode artifacts."""

from __future__ import annotations

from typing import Any


NONEMPTY_STRING = {"type": "string", "minLength": 1}
STRING_ARRAY = {"type": "array", "items": NONEMPTY_STRING, "uniqueItems": True}

GENERATION_REQUEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://memory-r1.local/schemas/mmem-v2-generation-request.schema.json",
    "title": "MMem-v2 model-backed generation request",
    "type": "object",
    "required": ["episode_id"],
    "properties": {
        "episode_id": NONEMPTY_STRING,
        "dataset": NONEMPTY_STRING,
        "language": {**NONEMPTY_STRING, "default": "en"},
        "session_count": {"type": "integer", "minimum": 1},
        "time_span_days": {"type": "integer", "minimum": 1},
        "rounds_per_session_min": {"type": "integer", "minimum": 1},
        "rounds_per_session_max": {"type": "integer", "minimum": 1},
        "images_per_session_min": {"type": "integer", "minimum": 0},
        "images_per_session_max": {"type": "integer", "minimum": 0},
        "qa_count": {"type": "integer", "minimum": 1},
        "task_ratios": {
            "type": "object",
            "propertyNames": {"enum": ["FR", "VS", "TTL", "TR", "VR", "MR", "KR", "CD", "AR"]},
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "scenario_constraints": STRING_ARRAY,
        "existing_cluster_summaries": STRING_ARRAY,
        "seed": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

EPISODE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://memory-r1.local/schemas/mmem-v2-episode.schema.json",
    "title": "MMem Base-v0 episode artifact",
    "type": "object",
    "required": [
        "schema_version", "dataset", "episode_id", "character_profile", "sessions",
        "images", "observed_facts", "qa_candidates",
    ],
    "properties": {
        "schema_version": {"const": "mmem-v2.0"},
        "dataset": NONEMPTY_STRING,
        "episode_id": NONEMPTY_STRING,
        "character_profile": {"type": "object"},
        "sessions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["session_id", "date", "events"],
                "properties": {
                    "session_id": NONEMPTY_STRING,
                    "date": NONEMPTY_STRING,
                    "events": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "event_id", "session_id", "turn_index", "timestamp", "user", "assistant", "image_ids"
                            ],
                            "properties": {
                                "event_id": NONEMPTY_STRING,
                                "session_id": NONEMPTY_STRING,
                                "turn_index": {"type": "integer", "minimum": 1},
                                "timestamp": {"type": "string"},
                                "user": NONEMPTY_STRING,
                                "assistant": NONEMPTY_STRING,
                                "image_ids": STRING_ARRAY,
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "image_id", "path", "public_retrieval_description", "private_verified_visual_facts"
                ],
                "properties": {
                    "image_id": NONEMPTY_STRING,
                    "path": NONEMPTY_STRING,
                    "role": {"enum": ["memory", "query", "both"]},
                    "public_retrieval_description": {"type": "string"},
                    "private_verified_visual_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["visual_fact_id", "predicate", "value"],
                            "properties": {
                                "visual_fact_id": NONEMPTY_STRING,
                                "subject": {"type": "string"},
                                "predicate": NONEMPTY_STRING,
                                "value": {},
                                "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                                "verifier_agreement": {"type": ["integer", "null"], "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "image_requests": {
            "type": "array",
            "description": "Construction-private GPT Image requests; materialized into images before validation.",
            "items": {
                "type": "object",
                "required": ["image_id", "prompt", "public_retrieval_description"],
                "properties": {
                    "image_id": NONEMPTY_STRING,
                    "prompt": NONEMPTY_STRING,
                    "public_retrieval_description": {"type": "string"},
                    "private_verified_visual_facts": {"type": "array"},
                    "role": {"enum": ["memory", "query", "both"]},
                },
                "additionalProperties": False,
            },
        },
        "observed_facts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "fact_id", "subject", "predicate", "object", "epistemic_status", "lifecycle_status",
                    "valid_from_session", "supersedes", "contradicts", "observed_provenance",
                ],
                "properties": {
                    "fact_id": NONEMPTY_STRING,
                    "subject": NONEMPTY_STRING,
                    "predicate": NONEMPTY_STRING,
                    "object": {},
                    "epistemic_status": NONEMPTY_STRING,
                    "lifecycle_status": {"enum": ["active", "superseded", "retracted"]},
                    "valid_from_session": NONEMPTY_STRING,
                    "valid_to_session": {"type": ["string", "null"]},
                    "supersedes": STRING_ARRAY,
                    "contradicts": STRING_ARRAY,
                    "observed_provenance": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["event_id", "text_spans", "visual_fact_ids"],
                            "properties": {
                                "event_id": NONEMPTY_STRING,
                                "text_spans": STRING_ARRAY,
                                "visual_fact_ids": STRING_ARRAY,
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "qa_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "qa_id", "task", "memory_cutoff", "question_modality", "question_text",
                    "question_image_ids", "answer", "answer_type", "task_oracle",
                ],
                "properties": {
                    "qa_id": NONEMPTY_STRING,
                    "task": {"enum": ["FR", "TR", "KR", "AR", "VS", "VR", "MR", "CD", "TTL"]},
                    "memory_cutoff": {
                        "type": "object",
                        "required": ["mode"],
                        "properties": {
                            "mode": {"enum": ["episode_end", "session_end", "event", "timestamp"]},
                            "session_id": {"type": "string"},
                            "event_id": {"type": "string"},
                            "timestamp": {"type": "string"},
                        },
                    },
                    "question_modality": {"enum": ["text", "image", "text+image"]},
                    "question_text": NONEMPTY_STRING,
                    "question_image_ids": STRING_ARRAY,
                    "answer": NONEMPTY_STRING,
                    "canonical_answer": {},
                    "answer_type": NONEMPTY_STRING,
                    "required_evidence_sets": {"type": "array", "items": STRING_ARRAY},
                    "required_visual_fact_ids": STRING_ARRAY,
                    "supporting_event_ids": STRING_ARRAY,
                    "hard_negatives": STRING_ARRAY,
                    "answer_function": {"type": "object"},
                    "task_oracle": {"type": "object", "required": ["kind"]},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": False,
}
