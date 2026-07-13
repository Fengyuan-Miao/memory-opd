#!/usr/bin/env python3
# Copyright 2025 Individual Contributor: Fengyuan Miao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare a verl FSDP actor checkpoint for OPD-MM vLLM evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "opd_mm_checkpoint_manifest.json"
KEY_PREFIX_REWRITES = (
    ("model.language_model.language_model.language_model.", "model.language_model."),
    ("model.language_model.visual.", "model.visual."),
)


def normalize_qwen35_key(key: str) -> str:
    """Remove the extra training-wrapper prefixes emitted by the FSDP merger."""
    for source, target in KEY_PREFIX_REWRITES:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def _weight_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def rewrite_qwen35_safetensor_keys(model_dir: Path) -> int:
    """Rewrite checkpoint keys in place and return the number of renamed tensors."""
    renamed_total = 0
    for path in _weight_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        renamed = sum(normalize_qwen35_key(key) != key for key in keys)
        if renamed == 0:
            continue

        state = load_file(path, device="cpu")
        normalized: dict[str, Any] = {}
        for key, tensor in state.items():
            normalized_key = normalize_qwen35_key(key)
            if normalized_key in normalized:
                raise ValueError(f"checkpoint key collision after normalization: {normalized_key}")
            normalized[normalized_key] = tensor

        temporary_path = path.with_name(f".{path.name}.rewrite-{os.getpid()}")
        save_file(normalized, temporary_path, metadata={"format": "pt"})
        temporary_path.replace(path)
        renamed_total += renamed

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") or {}
        normalized_map = {normalize_qwen35_key(key): value for key, value in weight_map.items()}
        if len(normalized_map) != len(weight_map):
            raise ValueError("checkpoint index key collision after normalization")
        index["weight_map"] = normalized_map
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return renamed_total


def validate_vllm_checkpoint(model_dir: Path) -> dict[str, Any]:
    """Fail closed unless the directory has the key layout expected by vLLM."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {model_dir}")
    weight_files = _weight_files(model_dir)
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found in {model_dir}")

    keys: set[str] = set()
    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        duplicate_keys = keys & shard_keys
        if duplicate_keys:
            raise ValueError(f"duplicate keys across checkpoint shards: {sorted(duplicate_keys)[:3]}")
        keys.update(shard_keys)

    unnormalized = [key for key in keys if normalize_qwen35_key(key) != key]
    if unnormalized:
        raise ValueError(f"unnormalized Qwen3.5 checkpoint keys remain: {unnormalized[:3]}")
    required_checks = {
        "language embeddings": "model.language_model.embed_tokens.weight" in keys,
        "language layers": any(key.startswith("model.language_model.layers.") for key in keys),
        "visual encoder": any(key.startswith("model.visual.") for key in keys),
    }
    missing = [name for name, present in required_checks.items() if not present]
    if missing:
        raise ValueError(f"checkpoint has an unexpected key layout; missing {missing}")
    return {
        "tensor_count": len(keys),
        "weight_files": [path.name for path in weight_files],
        "required_key_checks": required_checks,
    }


def prepare_checkpoint(checkpoint_dir: Path, output_dir: Path | None = None) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    actor_dir = checkpoint_dir / "actor"
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"missing actor checkpoint directory: {actor_dir}")
    output_dir = (output_dir or checkpoint_dir / "actor_merged_hf_vllm_fixed").expanduser().resolve()

    if output_dir.exists():
        validation = validate_vllm_checkpoint(output_dir)
        print(f"Reusing validated checkpoint: {output_dir}")
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return output_dir

    building_dir = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if building_dir.exists():
        raise FileExistsError(f"temporary checkpoint directory already exists: {building_dir}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(building_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        renamed_tensors = rewrite_qwen35_safetensor_keys(building_dir)
        validation = validate_vllm_checkpoint(building_dir)
        manifest = {
            "source_checkpoint_dir": str(checkpoint_dir),
            "source_actor_dir": str(actor_dir),
            "prepared_model_dir": str(output_dir),
            "renamed_tensors": renamed_tensors,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            **validation,
        }
        (building_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        building_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(building_dir, ignore_errors=True)
        raise

    print(f"Prepared checkpoint: {output_dir}")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_checkpoint(args.checkpoint_dir, args.output_dir)
    print(f"READY_MODEL_DIR={prepared}")


if __name__ == "__main__":
    main()
