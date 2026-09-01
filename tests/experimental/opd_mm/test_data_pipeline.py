"""Focused tests for the deterministic MMem Base-v0 pipeline."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from verl.experimental.opd_mm.data_pipeline.generation import (
    GenerationConfig,
    GenerationRequest,
    MMemGenerationPipeline,
)
from verl.experimental.opd_mm.data_pipeline.image_generation import materialize_image_requests
from verl.experimental.opd_mm.data_pipeline.multimodal_generation import (
    DEFAULT_MULTIMODAL_MODEL,
    MultimodalResponsesClient,
)
from verl.experimental.opd_mm.data_pipeline.pipeline import build_dataset, build_episode


def _episode() -> dict:
    return {
        "schema_version": "mmem-v2.0",
        "dataset": "mmem_v2",
        "episode_id": "EP_TEST",
        "character_profile": {
            "name": "Mira",
            "mutable_states": {"studio wall": "green"},
        },
        "sessions": [
            {
                "session_id": "S1",
                "date": "2026-01-01",
                "events": [
                    {
                        "event_id": "EV_S1_T1",
                        "session_id": "S1",
                        "turn_index": 1,
                        "timestamp": "2026-01-01T10:00:00",
                        "user": "I chose blue for the studio wall.",
                        "assistant": "Blue is now the recorded choice.",
                        "image_ids": [],
                    }
                ],
            },
            {
                "session_id": "S2",
                "date": "2026-02-01",
                "events": [
                    {
                        "event_id": "EV_S2_T1",
                        "session_id": "S2",
                        "turn_index": 1,
                        "timestamp": "2026-02-01T10:00:00",
                        "user": "I changed the studio wall choice to green.",
                        "assistant": "Green replaces the earlier choice.",
                        "image_ids": [],
                    }
                ],
            },
        ],
        "images": [],
        "observed_facts": [
            {
                "fact_id": "F_COLOR_BLUE",
                "subject": "studio.wall",
                "predicate": "paint.color",
                "object": "blue",
                "epistemic_status": "asserted",
                "lifecycle_status": "active",
                "valid_from_session": "S1",
                "valid_to_session": None,
                "supersedes": [],
                "contradicts": [],
                "observed_provenance": [
                    {"event_id": "EV_S1_T1", "text_spans": ["chose blue"], "visual_fact_ids": []}
                ],
            },
            {
                "fact_id": "F_COLOR_GREEN",
                "subject": "studio.wall",
                "predicate": "paint.color",
                "object": "green",
                "epistemic_status": "asserted",
                "lifecycle_status": "active",
                "valid_from_session": "S2",
                "valid_to_session": None,
                "supersedes": ["F_COLOR_BLUE"],
                "contradicts": ["F_COLOR_BLUE"],
                "observed_provenance": [
                    {"event_id": "EV_S2_T1", "text_spans": ["choice to green"], "visual_fact_ids": []}
                ],
            },
        ],
        "qa_candidates": [
            {
                "qa_id": "QA_FR",
                "task": "FR",
                "memory_cutoff": {"mode": "event", "event_id": "EV_S1_T1"},
                "question_modality": "text",
                "question_text": "What color had been selected for the studio wall at that point?",
                "question_image_ids": [],
                "answer": "blue",
                "answer_type": "short_text",
                "required_evidence_sets": [],
                "required_visual_fact_ids": [],
                "supporting_event_ids": [],
                "hard_negatives": [],
                "answer_function": {"kind": "fact_value"},
                "task_oracle": {"kind": "fact_lookup", "fact_id": "F_COLOR_BLUE"},
            },
            {
                "qa_id": "QA_TR",
                "task": "TR",
                "memory_cutoff": {"mode": "episode_end"},
                "question_modality": "text",
                "question_text": "Did the original selection happen before the later change?",
                "question_image_ids": [],
                "answer": "Yes",
                "answer_type": "boolean",
                "required_evidence_sets": [],
                "required_visual_fact_ids": [],
                "supporting_event_ids": [],
                "hard_negatives": [],
                "answer_function": {"kind": "boolean"},
                "task_oracle": {
                    "kind": "temporal_relation",
                    "left_event_id": "EV_S1_T1",
                    "right_event_id": "EV_S2_T1",
                    "relation": "before",
                },
            },
            {
                "qa_id": "QA_KR",
                "task": "KR",
                "memory_cutoff": {"mode": "episode_end"},
                "question_modality": "text",
                "question_text": "What is the current studio wall choice?",
                "question_image_ids": [],
                "answer": "green",
                "answer_type": "short_text",
                "required_evidence_sets": [],
                "required_visual_fact_ids": [],
                "supporting_event_ids": [],
                "hard_negatives": [],
                "answer_function": {"kind": "latest_value"},
                "task_oracle": {
                    "kind": "latest_valid_value",
                    "subject": "studio.wall",
                    "predicate": "paint.color",
                    "old_fact_id": "F_COLOR_BLUE",
                    "new_fact_id": "F_COLOR_GREEN",
                },
            },
            {
                "qa_id": "QA_AR",
                "task": "AR",
                "memory_cutoff": {"mode": "episode_end"},
                "question_modality": "text",
                "question_text": "What price was given for the studio paint?",
                "question_image_ids": [],
                "answer": "Not mentioned.",
                "answer_type": "absence",
                "required_evidence_sets": [],
                "required_visual_fact_ids": [],
                "supporting_event_ids": [],
                "hard_negatives": [],
                "answer_function": {"kind": "absence"},
                "task_oracle": {
                    "kind": "absence_in_closed_scope",
                    "subject": "studio.paint",
                    "closed_world_scope": ["paint.price"],
                    "missing_predicate": "paint.price",
                    "topic_anchor_event_ids": ["EV_S2_T1"],
                },
            },
        ],
    }


class DataPipelineTest(unittest.TestCase):
    def test_model_backed_generation_is_observed_only_and_deterministically_accepted(self) -> None:
        responses = [
            {
                "episode_id": "EP_GENERATED",
                "scenario": {
                    "primary_domains": ["home project"],
                    "secondary_domains": [],
                    "language": "en",
                    "time_span_days": 1,
                },
                "persona": {
                    "user_id": "U1",
                    "name": "Mira",
                    "occupation": "designer",
                    "communication_style": ["informal"],
                    "stable_traits": ["likes color studies"],
                    "mutable_states": {},
                },
                "session_plan": [
                    {
                        "session_id": "S1",
                        "date": "2026-09-01",
                        "main_event": "choose a studio wall color",
                        "callback_topics": [],
                        "target_rounds": 1,
                        "target_image_count": 0,
                    }
                ],
                "task_hooks": [],
                "recurring_entities": [],
            },
            {
                "planned_facts": [
                    {
                        "fact_id": "PLAN_F1",
                        "subject": "studio.wall",
                        "predicate": "paint.color",
                        "object": "blue",
                        "epistemic_status": "confirmed",
                        "lifecycle_status": "active",
                        "valid_from_session": "S1",
                        "valid_to_session": None,
                        "supersedes": [],
                        "contradicts": [],
                    }
                ],
                "events": [
                    {
                        "event_id": "EV1",
                        "session_id": "S1",
                        "turn_index": 1,
                        "timestamp": "2026-09-01T10:00:00",
                        "purpose": "record the choice",
                        "fact_ids_to_express": ["PLAN_F1"],
                        "image_ids": [],
                        "allowed_user_information": ["the selected wall color is blue"],
                        "ordinary_context": "studio renovation",
                    }
                ],
                "task_hooks": [],
                "image_needs": [],
            },
            {"user": "I chose blue for the studio wall.", "image_ids": []},
            {"assistant": "That should give the studio a calm look."},
            {
                "observed_facts": [
                    {
                        "fact_id": "F1",
                        "subject": "studio.wall",
                        "predicate": "paint.color",
                        "object": "blue",
                        "epistemic_status": "confirmed",
                        "lifecycle_status": "active",
                        "valid_from_session": "S1",
                        "valid_to_session": None,
                        "supersedes": [],
                        "contradicts": [],
                        "observed_provenance": [
                            {"event_id": "EV1", "text_spans": ["chose blue"], "visual_fact_ids": []}
                        ],
                    }
                ]
            },
            {
                "qa_specs": [
                    {
                        "qa_id": "QA1",
                        "task": "FR",
                        "memory_cutoff": {"mode": "episode_end"},
                        "question_intent": "retrieve the selected wall color",
                        "question_modality": "text",
                        "question_image_ids": [],
                        "answer": "blue",
                        "answer_type": "short_text",
                        "required_evidence_sets": [],
                        "required_visual_fact_ids": [],
                        "supporting_event_ids": [],
                        "hard_negatives": [],
                        "answer_function": {"kind": "fact_value"},
                        "task_oracle": {"kind": "fact_lookup", "required_fact_ids": ["F1"]},
                    }
                ]
            },
            {"candidates": [{"question_text": "Which color was selected for the studio wall?"}]},
            {
                "judgments": [
                    {
                        "qa_id": "QA1",
                        "accepted": True,
                        "selected_index": 0,
                        "error_codes": [],
                        "minimal_revision": "",
                    }
                ]
            },
        ]

        class FakeMultimodal:
            def generate_json(self, **_: object):
                return responses.pop(0), None

        class UnusedImageClient:
            def generate(self, *_: object, **__: object):
                raise AssertionError("image generation should not be called for an image-free episode")

        request = GenerationRequest.from_dict(
            {
                "episode_id": "EP_GENERATED",
                "language": "en",
                "session_count": 1,
                "rounds_per_session_min": 1,
                "rounds_per_session_max": 1,
                "images_per_session_min": 0,
                "images_per_session_max": 0,
                "qa_count": 1,
                "task_ratios": {"FR": 1.0},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = MMemGenerationPipeline(
                multimodal_client=FakeMultimodal(),
                image_client=UnusedImageClient(),
                config=GenerationConfig(stage_retries=0, run_full_state_audit=False),
            )
            generated = pipeline.generate(request, work_root=temporary)
            artifact = json.loads(generated.artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(generated.accepted_qa_count, 1)
        self.assertEqual(artifact["observed_facts"][0]["fact_id"], "F1")
        self.assertNotIn("planned_facts", artifact)
        self.assertFalse(responses)

    def test_oracles_and_event_evidence(self) -> None:
        result = build_episode(_episode(), strict=True)
        self.assertEqual(len(result.accepted_qas), 4)
        by_id = {qa["qa_id"]: qa for qa in result.accepted_qas}
        self.assertEqual(by_id["QA_FR"]["required_evidence_sets"], [["EV_S1_T1"]])
        self.assertEqual(by_id["QA_TR"]["required_evidence_sets"], [["EV_S1_T1", "EV_S2_T1"]])
        self.assertEqual(by_id["QA_KR"]["required_evidence_sets"], [["EV_S1_T1", "EV_S2_T1"]])
        self.assertEqual(by_id["QA_AR"]["required_evidence_sets"], [["EV_S2_T1"]])

    def test_bad_gold_is_rejected(self) -> None:
        value = _episode()
        value["qa_candidates"][0]["answer"] = "green"
        result = build_episode(value)
        self.assertEqual(len(result.rejected_qas), 1)
        self.assertEqual(result.certificates[0]["issues"][0]["code"], "oracle_failure")

    def test_image_request_is_private_materialized_and_resized(self) -> None:
        class FakeClient:
            def generate(self, prompt: str) -> bytes:
                self.prompt = prompt
                output = io.BytesIO()
                Image.new("RGB", (8, 8), "white").save(output, format="PNG")
                return output.getvalue()

        value = _episode()
        value["image_requests"] = [
            {
                "image_id": "IMG_1",
                "prompt": "private construction prompt",
                "public_retrieval_description": "A simple room.",
                "private_verified_visual_facts": [],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            materialized, generated = materialize_image_requests(
                value, output_dir=temporary, client=FakeClient()
            )
            self.assertNotIn("image_requests", materialized)
            self.assertNotIn("prompt", materialized["images"][0])
            image_path = Path(materialized["images"][0]["path"])
            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (512, 512))
            self.assertEqual(generated[0].path, image_path)

    def test_portable_export_and_opd_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "test.episode.json"
            artifact.write_text(json.dumps(_episode()), encoding="utf-8")
            output = root / "release"
            manifest = build_dataset([artifact], output_root=output, strict=True)
            self.assertEqual(manifest["accepted_qa_count"], 4)
            self.assertTrue((output / "data" / "dialog" / "EP_TEST.json").is_file())
            self.assertTrue((output / "opd_mm_store" / "records.jsonl").is_file())
            public_episode = json.loads((output / "data" / "dialog" / "EP_TEST.json").read_text())
            self.assertEqual(public_episode["character_profile"], {"name": "Mira"})
            self.assertFalse(public_episode["annotation_provenance"]["human_reviewed"])
            self.assertTrue(
                all(
                    turn.get("timestamp")
                    for session in public_episode["multi_session_dialogues"]
                    for turn in session["dialogues"]
                )
            )
            self.assertTrue((output / "reports" / "public_admission.json").is_file())
            qas = [json.loads(line) for line in (output / "opd_mm_store" / "qas.jsonl").read_text().splitlines()]
            self.assertEqual(len(qas), 4)
            self.assertEqual(qas[0]["dataset"], "mmem_v2")
            self.assertTrue(qas[0]["sample_id"].startswith("mmem_v2:EP_TEST:"))

    def test_multimodal_client_uses_terra_and_direct_task_instructions(self) -> None:
        class FakeResponse:
            output_text = '{"accepted":true}'

            def model_dump(self, **_: object) -> dict:
                return {
                    "id": "response_test",
                    "model": DEFAULT_MULTIMODAL_MODEL,
                    "usage": {"total_tokens": 7},
                    "output": [],
                }

        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs: object) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        class FakeOpenAI:
            instance = None

            def __init__(self, **_: object) -> None:
                self.responses = FakeResponses()
                FakeOpenAI.instance = self

            def close(self) -> None:
                pass

        with patch(
            "verl.experimental.opd_mm.data_pipeline.multimodal_generation.OpenAI",
            FakeOpenAI,
        ):
            with MultimodalResponsesClient(api_key="test", use_env_proxy=False) as client:
                value, result = client.generate_json(
                    task_contract="Return whether the input is accepted.",
                    prompt="Inspect this input.",
                )
        self.assertEqual(value, {"accepted": True})
        self.assertEqual(result.model, DEFAULT_MULTIMODAL_MODEL)
        kwargs = FakeOpenAI.instance.responses.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.6-terra")
        self.assertEqual(
            kwargs["instructions"],
            "Return whether the input is accepted.\n"
            "Return exactly one valid JSON object. Do not add Markdown or commentary.",
        )


if __name__ == "__main__":
    unittest.main()
