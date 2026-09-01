"""Mem-Gallery-style MMem Base-v0 construction pipeline.

The package deliberately separates generative stages from deterministic data
acceptance.  Upstream models produce an :class:`Episode`; this package owns
the observed fact ledger, task oracles, evidence mining, validation, and
release export.
"""

from .generation import GenerationConfig, GenerationRequest, GeneratedEpisode, MMemGenerationPipeline
from .models import Episode, MemoryCutoff, ObservedFact, QACandidate
from .multimodal_generation import (
    DEFAULT_MULTIMODAL_BASE_URL,
    DEFAULT_MULTIMODAL_MODEL,
    MultimodalResponsesClient,
)
from .pipeline import BuildResult, build_dataset, build_episode

__all__ = [
    "BuildResult",
    "DEFAULT_MULTIMODAL_BASE_URL",
    "DEFAULT_MULTIMODAL_MODEL",
    "Episode",
    "GeneratedEpisode",
    "GenerationConfig",
    "GenerationRequest",
    "MemoryCutoff",
    "MMemGenerationPipeline",
    "MultimodalResponsesClient",
    "ObservedFact",
    "QACandidate",
    "build_dataset",
    "build_episode",
]
