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

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.opd_mm_baseline.prepare_opd_mm_checkpoint import (
    normalize_qwen35_key,
    rewrite_qwen35_safetensor_keys,
    validate_vllm_checkpoint,
)


def test_qwen35_checkpoint_key_rewrite_and_validation(tmp_path) -> None:
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    save_file(
        {
            "model.language_model.language_model.language_model.embed_tokens.weight": torch.zeros(2, 2),
            "model.language_model.language_model.language_model.layers.0.input_layernorm.weight": torch.ones(2),
            "model.language_model.visual.pos_embed.weight": torch.zeros(2, 2),
        },
        model_dir / "model.safetensors",
    )

    assert rewrite_qwen35_safetensor_keys(model_dir) == 3
    validation = validate_vllm_checkpoint(model_dir)

    assert validation["tensor_count"] == 3
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.visual.pos_embed.weight",
        }


def test_qwen35_checkpoint_key_normalization_leaves_fixed_keys_unchanged() -> None:
    key = "model.language_model.layers.0.input_layernorm.weight"
    assert normalize_qwen35_key(key) == key
