#!/usr/bin/env python3
"""Build MMem-v2 artifacts, optionally materializing GPT Image requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.data_pipeline.image_generation import (
    GPTImageClient,
    materialize_image_requests,
)
from verl.experimental.opd_mm.data_pipeline.pipeline import build_dataset
from verl.experimental.opd_mm.data_pipeline.schema import EPISODE_SCHEMA


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _inputs(values: list[str]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            result.extend(
                candidate
                for candidate in sorted(path.rglob("*.json"))
                if candidate.name.endswith(("episode.json", ".episode.json"))
            )
        else:
            result.append(path)
    if not result:
        raise ValueError("no episode artifacts found")
    return [path.resolve() for path in result]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="Episode JSON file or directory; repeatable")
    parser.add_argument("--output", default="dataset/mmem_v2", help="Portable dataset output root")
    parser.add_argument("--strict", action="store_true", help="Fail if any QA candidate is rejected")
    parser.add_argument("--skip-image-generation", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--image-workers", type=int, default=1)
    parser.add_argument("--write-schema", help="Write the canonical JSON Schema and exit")
    args = parser.parse_args()

    if args.write_schema:
        path = Path(args.write_schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(EPISODE_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.resolve())
        return
    if not args.input:
        parser.error("--input is required unless --write-schema is used")

    paths = _inputs(args.input)
    prepared: dict[str, dict[str, Any]] = {}
    client: GPTImageClient | None = None
    output = Path(args.output).resolve()
    for path in paths:
        value = _read(path)
        requests = value.get("image_requests")
        if requests and not args.skip_image_generation:
            if client is None:
                client = GPTImageClient()
            episode_id = str(value.get("episode_id") or path.stem)
            value, generated = materialize_image_requests(
                value,
                output_dir=output / "generated_images" / episode_id,
                client=client,
                workers=max(args.image_workers, 1),
                overwrite=args.overwrite_images,
            )
            print(f"{episode_id}: materialized {len(generated)} GPT Image image(s)")
        elif requests:
            raise ValueError(f"{path} contains image_requests but image generation was disabled")
        prepared[str(path)] = value
    manifest = build_dataset(paths, output_root=output, strict=args.strict, prepared_artifacts=prepared)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
