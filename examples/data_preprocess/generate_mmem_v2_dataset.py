#!/usr/bin/env python3
"""Generate MMem-v2 episodes with Terra/GPT Image, then run deterministic acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.data_pipeline.generation import (
    GenerationConfig,
    GenerationRequest,
    MMemGenerationPipeline,
)
from verl.experimental.opd_mm.data_pipeline.image_generation import GPTImageClient
from verl.experimental.opd_mm.data_pipeline.multimodal_generation import (
    DEFAULT_MULTIMODAL_BASE_URL,
    DEFAULT_MULTIMODAL_MODEL,
    MultimodalResponsesClient,
)
from verl.experimental.opd_mm.data_pipeline.pipeline import build_dataset
from verl.experimental.opd_mm.data_pipeline.schema import GENERATION_REQUEST_SCHEMA


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", action="append", default=[], help="Generation request JSON; repeatable")
    parser.add_argument("--output", default="dataset/mmem_v2_generated", help="Portable accepted dataset root")
    parser.add_argument("--work-dir", default=None, help="Private construction artifacts; defaults under output")
    parser.add_argument("--multimodal-base-url", default=DEFAULT_MULTIMODAL_BASE_URL)
    parser.add_argument("--multimodal-model", default=DEFAULT_MULTIMODAL_MODEL)
    parser.add_argument("--no-model-env-proxy", action="store_true")
    parser.add_argument("--stage-retries", type=int, default=2)
    parser.add_argument("--image-candidates", type=int, default=1)
    parser.add_argument("--image-repair-rounds", type=int, default=1)
    parser.add_argument("--qa-paraphrase-candidates", type=int, default=2)
    parser.add_argument("--run-full-state-audit", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible private plans from work-dir")
    parser.add_argument("--write-request-schema", help="Write the generation-request JSON Schema and exit")
    args = parser.parse_args()

    if args.write_request_schema:
        schema_path = Path(args.write_request_schema)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            json.dumps(GENERATION_REQUEST_SCHEMA, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(schema_path.resolve())
        return
    if not args.request:
        parser.error("--request is required unless --write-request-schema is used")

    output = Path(args.output).resolve()
    work_root = Path(args.work_dir).resolve() if args.work_dir else output / ".construction"
    requests = [GenerationRequest.from_dict(_read(Path(value).resolve())) for value in args.request]
    config = GenerationConfig(
        stage_retries=max(args.stage_retries, 0),
        image_candidates=max(args.image_candidates, 1),
        image_repair_rounds=max(args.image_repair_rounds, 0),
        qa_paraphrase_candidates=max(args.qa_paraphrase_candidates, 1),
        run_full_state_audit=args.run_full_state_audit,
        overwrite_images=args.overwrite_images,
        resume=args.resume,
    )
    image_client = GPTImageClient()
    with MultimodalResponsesClient(
        base_url=args.multimodal_base_url,
        model=args.multimodal_model,
        use_env_proxy=not args.no_model_env_proxy,
    ) as multimodal:
        generator = MMemGenerationPipeline(
            multimodal_client=multimodal,
            image_client=image_client,
            config=config,
        )
        generated = [generator.generate(request, work_root=work_root) for request in requests]
    artifact_paths = [item.artifact_path for item in generated]
    manifest = build_dataset(artifact_paths, output_root=output, strict=True)
    manifest["generation"] = [
        {
            "episode_id": request.episode_id,
            "work_dir": str(result.work_dir),
            "accepted_qa_count": result.accepted_qa_count,
            "rejected_qa_count": result.rejected_qa_count,
        }
        for request, result in zip(requests, generated, strict=True)
    ]
    manifest_path = output / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
