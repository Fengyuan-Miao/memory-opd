#!/usr/bin/env python3
# Copyright 2025 Individual Contributor: Fengyuan Miao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare a fixed Mem-Gallery QA JSONL as a verl validation parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.data_preprocess.build_mem_gallery_opd_mm_train_subset import _samples_for_qas
from verl.experimental.opd_mm.dataset import write_opd_rlhf_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qas-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", default="dataset/mem_gallery")
    parser.add_argument("--data-source", default="opd_mm_eval")
    parser.add_argument("--agent-name", default="tool_agent")
    args = parser.parse_args()

    qas = [
        json.loads(line)
        for line in Path(args.qas_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = _samples_for_qas(
        qas,
        dataset_root=args.dataset_root,
        data_source=args.data_source,
        agent_name=args.agent_name,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_opd_rlhf_parquet(
        samples,
        output,
        data_source=args.data_source,
        agent_name=args.agent_name,
    )
    print(json.dumps({"output": str(output.resolve()), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
