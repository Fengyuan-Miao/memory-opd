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

"""Disk-backed vector indexes for OPD-MM memory stores."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from verl.experimental.opd_mm.mem_gallery import (
    load_mem_gallery_qas,
    load_mem_gallery_records,
    memory_records_to_jsonl,
    qas_to_jsonl,
)
from verl.experimental.opd_mm.models import MemoryRecord
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, query_variants


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                rows.append(json.loads(value))
    return rows


def _write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return output


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return output


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float32")
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 0 else array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def _batched(values: list[Any], batch_size: int) -> Iterator[list[Any]]:
    size = max(1, int(batch_size))
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _progress(values: Iterable[list[Any]], total: int, desc: str) -> Iterable[list[Any]]:
    try:
        from tqdm.auto import tqdm

        return tqdm(values, total=total, desc=desc)
    except Exception:
        return values


def _records_from_jsonl(path: str | Path) -> list[MemoryRecord]:
    records = []
    for index, value in enumerate(_read_jsonl(path)):
        metadata = dict(value.get("metadata") or {})
        records.append(
            MemoryRecord(
                memory_id=str(value.get("memory_id", f"memory_{index}")),
                turn_id=str(value.get("turn_id", "")),
                timestamp=str(value.get("timestamp", "")),
                author=str(value.get("author", "")),
                modality=str(value.get("modality", "")),
                source_type=str(value.get("source_type", "")),
                summary=str(value.get("summary") or ""),
                content=str(value.get("content") or ""),
                raw_pointer=value.get("raw_pointer"),
                status=str(value.get("status", "active")),
                metadata=metadata,
            )
        )
    return records


@dataclass
class DiskVectorIndex:
    """A normalized numpy vector matrix plus one metadata row per vector."""

    index_dir: Path
    name: str
    embeddings: np.ndarray
    items: list[dict[str, Any]]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, index_dir: str | Path) -> "DiskVectorIndex":
        path = Path(index_dir)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        embeddings = np.load(path / "embeddings.npy", mmap_mode="r")
        items = _read_jsonl(path / "items.jsonl")
        return cls(
            index_dir=path,
            name=str(manifest.get("name") or path.name),
            embeddings=embeddings,
            items=items,
            manifest=manifest,
        )

    def __post_init__(self) -> None:
        self._row_by_memory_id = {
            str(item.get("memory_id")): index
            for index, item in enumerate(self.items)
        }

    def vector(self, memory_id: str) -> Optional[np.ndarray]:
        row = self._row_by_memory_id.get(memory_id)
        if row is None:
            return None
        return np.asarray(self.embeddings[row], dtype="float32")

    def search(
        self,
        query_vector: np.ndarray,
        memory_ids: Optional[set[str]] = None,
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        vector = _normalize(query_vector)
        if memory_ids is None:
            candidate_rows = np.arange(len(self.items))
        else:
            candidate_rows = np.asarray(
                [
                    row
                    for memory_id, row in self._row_by_memory_id.items()
                    if memory_id in memory_ids
                ],
                dtype=np.int64,
            )
        if candidate_rows.size == 0:
            return []
        scores = np.asarray(self.embeddings[candidate_rows] @ vector, dtype="float32")
        order = np.argsort(-scores)[: max(1, int(top_k))]
        return [
            (self.items[int(candidate_rows[row])], float(scores[row]))
            for row in order
        ]


class MiniLMTextEncoder:
    """Mean-pooled all-MiniLM-L6-v2 encoder using transformers only."""

    def __init__(self, model_path: str | Path, device: str = "cuda:0", max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text], batch_size=1)[0].tolist()

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        torch = self.torch
        with torch.no_grad():
            batches = list(_batched([str(text or "") for text in texts], batch_size))
            for batch in _progress(batches, len(batches), "dense"):
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                model_output = self.model(**inputs)
                token_embeddings = model_output.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().float().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 384), dtype="float32")


class SigLIPVisionEncoder:
    """SigLIP text/image encoder for text-to-image retrieval."""

    def __init__(self, model_path: str | Path, device: str = "cuda:0"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_images(paths: list[Any]) -> list[Any]:
        from PIL import Image

        return [Image.open(path).convert("RGB") for path in paths]

    @staticmethod
    def _feature_tensor(value: Any) -> Any:
        if hasattr(value, "pooler_output") and value.pooler_output is not None:
            return value.pooler_output
        return value

    def encode_image(self, image: Any) -> list[float]:
        return self.encode_images([image], batch_size=1)[0].tolist()

    def encode_images(self, images: list[Any], batch_size: int = 16) -> np.ndarray:
        outputs = []
        torch = self.torch
        with torch.no_grad():
            batches = list(_batched(images, batch_size))
            for batch in _progress(batches, len(batches), "vision-images"):
                pil_images = self._load_images(batch)
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                embeddings = self._feature_tensor(
                    self.model.get_image_features(**inputs)
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().float().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype="float32")

    def encode_text(self, text: str) -> list[float]:
        return self.encode_texts([text], batch_size=1)[0].tolist()

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        torch = self.torch
        with torch.no_grad():
            batches = list(_batched([str(text or "") for text in texts], batch_size))
            for batch in _progress(batches, len(batches), "vision-text"):
                inputs = self.processor(
                    text=batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                embeddings = self._feature_tensor(
                    self.model.get_text_features(**inputs)
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().float().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype="float32")


class GMEQwen2VLUnifiedEncoder:
    """GME-Qwen2-VL encoder compatible with transformers 5.x."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        min_image_tokens: int = 256,
        max_image_tokens: int = 512,
        max_length: int = 1024,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.max_length = int(max_length)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        min_pixels = int(min_image_tokens) * 28 * 28
        max_pixels = int(max_image_tokens) * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.processor.tokenizer.padding_side = "right"
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.default_instruction = "You are a helpful assistant."

    @staticmethod
    def _load_images(paths: list[Any]) -> list[Any]:
        from PIL import Image

        return [Image.open(path).convert("RGB") for path in paths]

    def _format_message(self, text: str | None, has_image: bool) -> str:
        content = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
        if text:
            content += str(text)
        return (
            f"<|im_start|>system\n{self.default_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n<|endoftext|>"
        )

    def encode_text(self, text: str) -> list[float]:
        return self.encode_texts([text], batch_size=1)[0].tolist()

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        messages = [self._format_message(text, has_image=False) for text in texts]
        return self._encode(messages, None, batch_size, desc="hybrid-text")

    def encode_image(self, image: Any, text: Optional[str] = None) -> list[float]:
        return self.encode_text_images([text or ""], [image], batch_size=1)[0].tolist()

    def encode_text_image(self, text: str, image: Any) -> list[float]:
        return self.encode_text_images([text], [image], batch_size=1)[0].tolist()

    def encode_text_images(
        self,
        texts: list[str],
        images: list[Any],
        batch_size: int = 4,
    ) -> np.ndarray:
        messages = [
            self._format_message(text, has_image=True)
            for text in texts
        ]
        return self._encode(messages, images, batch_size, desc="hybrid-images")

    def _encode(
        self,
        messages: list[str],
        images: Optional[list[Any]],
        batch_size: int,
        desc: str,
    ) -> np.ndarray:
        outputs = []
        torch = self.torch
        image_batches = (
            list(_batched(images, batch_size))
            if images is not None
            else [None] * ((len(messages) + batch_size - 1) // batch_size)
        )
        text_batches = list(_batched(messages, batch_size))
        with torch.no_grad():
            for text_batch, image_batch in _progress(
                list(zip(text_batches, image_batches)),
                len(text_batches),
                desc,
            ):
                pil_images = self._load_images(image_batch) if image_batch else None
                inputs = self.processor(
                    text=text_batch,
                    images=pil_images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                model_output = self.model.model(**inputs)
                sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
                embeddings = model_output.last_hidden_state[
                    torch.arange(model_output.last_hidden_state.shape[0], device=self.device),
                    sequence_lengths,
                ]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().float().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1536), dtype="float32")


class DiskIndexedHiddenMemoryStore(HiddenMemoryStore):
    """HiddenMemoryStore that reads memory vectors from disk indexes."""

    def __init__(
        self,
        records: Iterable[MemoryRecord],
        dense_index: Optional[DiskVectorIndex] = None,
        vision_index: Optional[DiskVectorIndex] = None,
        hybrid_index: Optional[DiskVectorIndex] = None,
        dense_query_encoder: Optional[MiniLMTextEncoder] = None,
        vision_query_encoder: Optional[SigLIPVisionEncoder] = None,
        hybrid_query_encoder: Optional[GMEQwen2VLUnifiedEncoder] = None,
    ):
        super().__init__(records)
        self.dense_index = dense_index
        self.vision_index = vision_index
        self.hybrid_index = hybrid_index
        self.dense_query_encoder = dense_query_encoder
        self.vision_query_encoder = vision_query_encoder
        self.hybrid_query_encoder = hybrid_query_encoder

    def dense_vector(self, record: MemoryRecord) -> Optional[np.ndarray]:
        if self.dense_index is None:
            return None
        return self.dense_index.vector(record.memory_id)

    def query_vectors(self, query: str) -> list[np.ndarray]:
        if self.dense_query_encoder is None:
            return []
        variants = query_variants(query)
        if not variants:
            return []
        return [
            np.asarray(vector, dtype="float32")
            for vector in self.dense_query_encoder.encode_batch(variants)
        ]

    def vision_vector(self, record: MemoryRecord) -> Optional[np.ndarray]:
        if self.vision_index is None:
            return None
        return self.vision_index.vector(record.memory_id)

    def vision_query_vector(
        self,
        query: str,
        question_image: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if self.vision_query_encoder is None:
            return None
        values = (
            self.vision_query_encoder.encode_image(question_image)
            if question_image
            else self.vision_query_encoder.encode_text(query)
        )
        return _normalize(np.asarray(values, dtype="float32"))

    def hybrid_vector(self, record: MemoryRecord) -> Optional[np.ndarray]:
        if self.hybrid_index is None:
            return None
        return self.hybrid_index.vector(record.memory_id)

    def hybrid_query_vector(
        self,
        query: str,
        question_image: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if self.hybrid_query_encoder is None:
            return None
        values = (
            self.hybrid_query_encoder.encode_text_image(query, question_image)
            if question_image
            else self.hybrid_query_encoder.encode_text(query)
        )
        return _normalize(np.asarray(values, dtype="float32"))


def _record_item(record: MemoryRecord, index_text: str, index_type: str) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "turn_id": record.turn_id,
        "timestamp": record.timestamp,
        "modality": record.modality,
        "source_type": record.source_type,
        "raw_pointer": record.raw_pointer,
        "index_text": index_text,
        "index_type": index_type,
        "scenario": record.metadata.get("scenario"),
        "session_id": record.metadata.get("session_id"),
        "round_id": record.metadata.get("round_id"),
        "image_id": record.metadata.get("image_id"),
    }


def _write_index(
    *,
    index_dir: Path,
    name: str,
    embeddings: np.ndarray,
    items: list[dict[str, Any]],
    model_path: str | Path,
    encoder: str,
) -> dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _normalize(embeddings)
    np.save(index_dir / "embeddings.npy", embeddings.astype("float32"))
    _write_jsonl(index_dir / "items.jsonl", items)
    manifest = {
        "name": name,
        "status": "complete",
        "encoder": encoder,
        "model_path": str(model_path),
        "metric": "cosine",
        "normalized": True,
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "created_at": _now(),
    }
    _write_json(index_dir / "manifest.json", manifest)
    return manifest


def _write_failed_index(index_dir: Path, name: str, error: BaseException) -> dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "created_at": _now(),
    }
    _write_json(index_dir / "manifest.json", manifest)
    return manifest


def _write_tables(
    records: list[MemoryRecord],
    qas: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    memory_records_to_jsonl(records, output_dir / "records.jsonl")
    qas_to_jsonl(qas, output_dir / "qas.jsonl")
    try:
        import pandas as pd

        pd.DataFrame(
            [record.to_dict(include_internal_id=True) for record in records]
        ).to_parquet(output_dir / "records.parquet", index=False)
        pd.DataFrame(qas).to_parquet(output_dir / "qas.parquet", index=False)
    except Exception as exc:
        _write_json(
            output_dir / "parquet_error.json",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )


def _build_vector_store(
    *,
    records: list[MemoryRecord],
    qas: list[dict[str, Any]],
    output_dir: str | Path,
    dataset_manifest: dict[str, Any],
    dense_model_path: str | Path,
    vision_model_path: str | Path,
    hybrid_model_path: str | Path,
    device: str,
    dense_batch_size: int,
    vision_batch_size: int,
    hybrid_text_batch_size: int,
    hybrid_image_batch_size: int,
    build_dense: bool,
    build_vision: bool,
    build_hybrid: bool,
) -> dict[str, Any]:
    """Persist records/QA tables and build the standard OPD-MM indexes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_tables(records, qas, output)
    manifest: dict[str, Any] = {
        **dataset_manifest,
        "output_dir": str(output.resolve()),
        "created_at": _now(),
        "record_count": len(records),
        "qa_count": len(qas),
        "indexes": {},
    }
    for index_name in ("dense", "vision", "hybrid"):
        index_manifest = output / "indexes" / index_name / "manifest.json"
        if index_manifest.exists():
            manifest["indexes"][index_name] = json.loads(
                index_manifest.read_text(encoding="utf-8")
            )

    if build_dense:
        try:
            encoder = MiniLMTextEncoder(dense_model_path, device=device)
            dense_records = [record for record in records if record.searchable_text()]
            dense_texts = [record.searchable_text() for record in dense_records]
            dense_items = [
                _record_item(record, text, "dense_text")
                for record, text in zip(dense_records, dense_texts)
            ]
            dense_embeddings = encoder.encode_batch(
                dense_texts,
                batch_size=dense_batch_size,
            )
            manifest["indexes"]["dense"] = _write_index(
                index_dir=output / "indexes" / "dense",
                name="dense",
                embeddings=dense_embeddings,
                items=dense_items,
                model_path=dense_model_path,
                encoder=type(encoder).__name__,
            )
            del encoder
        except Exception as exc:
            manifest["indexes"]["dense"] = _write_failed_index(
                output / "indexes" / "dense",
                "dense",
                exc,
            )

    if build_vision:
        try:
            encoder = SigLIPVisionEncoder(vision_model_path, device=device)
            vision_records = [
                record
                for record in records
                if record.raw_pointer and Path(record.raw_pointer).exists()
            ]
            vision_paths = [str(record.raw_pointer) for record in vision_records]
            vision_items = [
                _record_item(record, record.searchable_text(), "vision_image")
                for record in vision_records
            ]
            vision_embeddings = encoder.encode_images(
                vision_paths,
                batch_size=vision_batch_size,
            )
            manifest["indexes"]["vision"] = _write_index(
                index_dir=output / "indexes" / "vision",
                name="vision",
                embeddings=vision_embeddings,
                items=vision_items,
                model_path=vision_model_path,
                encoder=type(encoder).__name__,
            )
            del encoder
        except Exception as exc:
            manifest["indexes"]["vision"] = _write_failed_index(
                output / "indexes" / "vision",
                "vision",
                exc,
            )

    if build_hybrid:
        try:
            encoder = GMEQwen2VLUnifiedEncoder(hybrid_model_path, device=device)
            text_records = [
                record
                for record in records
                if record.searchable_text() and not record.raw_pointer
            ]
            image_records = [
                record
                for record in records
                if record.raw_pointer and Path(record.raw_pointer).exists()
            ]
            text_embeddings = encoder.encode_texts(
                [record.searchable_text() for record in text_records],
                batch_size=hybrid_text_batch_size,
            )
            image_embeddings = encoder.encode_text_images(
                [record.searchable_text() for record in image_records],
                [str(record.raw_pointer) for record in image_records],
                batch_size=hybrid_image_batch_size,
            )
            hybrid_records = text_records + image_records
            hybrid_items = [
                _record_item(record, record.searchable_text(), "hybrid_unified")
                for record in hybrid_records
            ]
            hybrid_embeddings = (
                np.concatenate([text_embeddings, image_embeddings], axis=0)
                if len(image_records)
                else text_embeddings
            )
            manifest["indexes"]["hybrid"] = _write_index(
                index_dir=output / "indexes" / "hybrid",
                name="hybrid",
                embeddings=hybrid_embeddings,
                items=hybrid_items,
                model_path=hybrid_model_path,
                encoder=type(encoder).__name__,
            )
            del encoder
        except Exception as exc:
            manifest["indexes"]["hybrid"] = _write_failed_index(
                output / "indexes" / "hybrid",
                "hybrid",
                exc,
            )

    _write_json(output / "manifest.json", manifest)
    return manifest


def build_mem_gallery_vector_store(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    dense_model_path: str | Path,
    vision_model_path: str | Path,
    hybrid_model_path: str | Path,
    device: str = "cuda:0",
    dense_batch_size: int = 128,
    vision_batch_size: int = 32,
    hybrid_text_batch_size: int = 16,
    hybrid_image_batch_size: int = 4,
    build_dense: bool = True,
    build_vision: bool = True,
    build_hybrid: bool = True,
) -> dict[str, Any]:
    """Build and persist a Mem-Gallery OPD-MM memory store."""
    records = load_mem_gallery_records(dataset_root)
    qas = load_mem_gallery_qas(dataset_root)
    return _build_vector_store(
        records=records,
        qas=qas,
        output_dir=output_dir,
        dataset_manifest={
            "dataset": "mem_gallery",
            "dataset_root": str(Path(dataset_root).resolve()),
        },
        dense_model_path=dense_model_path,
        vision_model_path=vision_model_path,
        hybrid_model_path=hybrid_model_path,
        device=device,
        dense_batch_size=dense_batch_size,
        vision_batch_size=vision_batch_size,
        hybrid_text_batch_size=hybrid_text_batch_size,
        hybrid_image_batch_size=hybrid_image_batch_size,
        build_dense=build_dense,
        build_vision=build_vision,
        build_hybrid=build_hybrid,
    )


def build_jsonl_vector_store(
    *,
    records_path: str | Path,
    qa_paths: Iterable[str | Path],
    output_dir: str | Path,
    dataset_name: str,
    dense_model_path: str | Path,
    vision_model_path: str | Path,
    hybrid_model_path: str | Path,
    device: str = "cuda:0",
    dense_batch_size: int = 128,
    vision_batch_size: int = 32,
    hybrid_text_batch_size: int = 16,
    hybrid_image_batch_size: int = 4,
    build_dense: bool = True,
    build_vision: bool = True,
    build_hybrid: bool = True,
) -> dict[str, Any]:
    """Build the Mem-Gallery-compatible indexes for an existing JSONL store."""

    resolved_qa_paths = [Path(path) for path in qa_paths]
    records = _records_from_jsonl(records_path)
    qas = [
        row
        for path in resolved_qa_paths
        for row in _read_jsonl(path)
    ]
    return _build_vector_store(
        records=records,
        qas=qas,
        output_dir=output_dir,
        dataset_manifest={
            "dataset": dataset_name,
            "records_source": str(Path(records_path).resolve()),
            "qa_sources": [str(path.resolve()) for path in resolved_qa_paths],
        },
        dense_model_path=dense_model_path,
        vision_model_path=vision_model_path,
        hybrid_model_path=hybrid_model_path,
        device=device,
        dense_batch_size=dense_batch_size,
        vision_batch_size=vision_batch_size,
        hybrid_text_batch_size=hybrid_text_batch_size,
        hybrid_image_batch_size=hybrid_image_batch_size,
        build_dense=build_dense,
        build_vision=build_vision,
        build_hybrid=build_hybrid,
    )


def load_indexed_memory_store(
    store_dir: str | Path,
    *,
    dense_model_path: str | Path | None = None,
    vision_model_path: str | Path | None = None,
    hybrid_model_path: str | Path | None = None,
    device: str = "cuda:0",
) -> DiskIndexedHiddenMemoryStore:
    """Load a disk-backed OPD-MM memory store and optional query encoders."""
    root = Path(store_dir)
    records = _records_from_jsonl(root / "records.jsonl")

    def load_index(name: str) -> Optional[DiskVectorIndex]:
        path = root / "indexes" / name
        if not (path / "embeddings.npy").exists():
            return None
        return DiskVectorIndex.load(path)

    dense_encoder = MiniLMTextEncoder(dense_model_path, device=device) if dense_model_path else None
    vision_encoder = SigLIPVisionEncoder(vision_model_path, device=device) if vision_model_path else None
    hybrid_encoder = (
        GMEQwen2VLUnifiedEncoder(hybrid_model_path, device=device)
        if hybrid_model_path
        else None
    )
    return DiskIndexedHiddenMemoryStore(
        records,
        dense_index=load_index("dense"),
        vision_index=load_index("vision"),
        hybrid_index=load_index("hybrid"),
        dense_query_encoder=dense_encoder,
        vision_query_encoder=vision_encoder,
        hybrid_query_encoder=hybrid_encoder,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the OPD-MM Mem-Gallery disk memory/vector store.",
    )
    parser.add_argument("--dataset-root", default="dataset/mem_gallery")
    parser.add_argument(
        "--output-dir",
        default="dataset/mem_gallery/opd_mm_store",
    )
    parser.add_argument(
        "--dense-model-path",
        default="/home/miaofy/data/pretrained_models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--vision-model-path",
        default="/home/miaofy/data/pretrained_models/SigLIP-Base-Patch16-384",
    )
    parser.add_argument(
        "--hybrid-model-path",
        default="/home/miaofy/data/pretrained_models/gme-Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-batch-size", type=int, default=128)
    parser.add_argument("--vision-batch-size", type=int, default=32)
    parser.add_argument("--hybrid-text-batch-size", type=int, default=16)
    parser.add_argument("--hybrid-image-batch-size", type=int, default=4)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_mem_gallery_vector_store(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dense_model_path=args.dense_model_path,
        vision_model_path=args.vision_model_path,
        hybrid_model_path=args.hybrid_model_path,
        device=args.device,
        dense_batch_size=args.dense_batch_size,
        vision_batch_size=args.vision_batch_size,
        hybrid_text_batch_size=args.hybrid_text_batch_size,
        hybrid_image_batch_size=args.hybrid_image_batch_size,
        build_dense=not args.skip_dense,
        build_vision=not args.skip_vision,
        build_hybrid=not args.skip_hybrid,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
