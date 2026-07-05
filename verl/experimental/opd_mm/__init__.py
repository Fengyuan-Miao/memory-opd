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

"""Experimental OPD-MM memory-retrieval environment for verl agent loops."""

from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.dataset import OPD_MM_SYSTEM_PROMPT, opd_messages_for_query, opd_sample_to_rlhf_record
from verl.experimental.opd_mm.models import (
    EvidenceItem,
    ExecutionResult,
    ExecutionStep,
    MemoryRecord,
    OPDRollout,
    OPDSample,
    PolicyOutput,
    PoolItem,
    SFTExample,
    ToolAction,
)
from verl.experimental.opd_mm.online_self_distill import maybe_collect_online_step_corrections
from verl.experimental.opd_mm.on_policy_distiller import OnPolicyDistiller
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, HybridRetriever, TurnAwareHybridRetriever
from verl.experimental.opd_mm.schema import TrajectoryValidationError, TrajectoryValidator
from verl.experimental.opd_mm.step_correction import StepCorrection, StepCorrectionCollector
from verl.experimental.opd_mm.vector_index import (
    DiskIndexedHiddenMemoryStore,
    DiskVectorIndex,
    load_indexed_memory_store,
)

__all__ = [
    "EvidenceItem",
    "DiskIndexedHiddenMemoryStore",
    "DiskVectorIndex",
    "ExecutionResult",
    "ExecutionStep",
    "HiddenMemoryStore",
    "HybridRetriever",
    "MemoryRecord",
    "OPDRollout",
    "OPDSample",
    "OnPolicyDistiller",
    "OPD_MM_SYSTEM_PROMPT",
    "PolicyOutput",
    "PoolItem",
    "SFTExample",
    "StepCorrection",
    "StepCorrectionCollector",
    "ToolAction",
    "ToolExecutor",
    "TrajectoryValidationError",
    "TrajectoryValidator",
    "TurnAwareHybridRetriever",
    "maybe_collect_online_step_corrections",
    "load_indexed_memory_store",
    "opd_messages_for_query",
    "opd_sample_to_rlhf_record",
]
