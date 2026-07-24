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

"""Generate and validate grounded STARK Mem-Gallery-style QA through an API."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.stark_expansion import (
    DeterministicQAClient,
    GPTQAClient,
    generate_qa_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-dir", default="dataset/Stark/opd_mm_store/expansion_v1")
    parser.add_argument("--records", default="dataset/Stark/opd_mm_store/records.jsonl")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--backend", choices=("api", "template"), default="api")
    parser.add_argument("--base-url", default="https://api.nideyiyi.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--api-key-env", default="STARK_LLM_API_KEY")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--circuit-breaker-errors", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    expansion_dir = Path(args.expansion_dir)
    output_dir = Path(args.output_dir) if args.output_dir else expansion_dir
    if args.backend == "template":
        client = DeterministicQAClient()
    else:
        api_key = os.getenv(args.api_key_env) or getpass.getpass("API key: ")
        if not api_key:
            raise ValueError("API key is required")
        client = GPTQAClient(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    summary = generate_qa_split(
        target_path=expansion_dir / "targets" / f"{args.split}.jsonl",
        records_path=args.records,
        output_dir=output_dir,
        split=args.split,
        client=client,
        max_episodes=args.max_episodes,
        resume=not args.no_resume,
        workers=args.workers,
        circuit_breaker_errors=args.circuit_breaker_errors,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
