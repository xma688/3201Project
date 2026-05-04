from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mast3r_backend import Mast3RBackend, Mast3RMatchResult


class Mast3RFeatureConfidence:
    def __init__(self, backend: Mast3RBackend) -> None:
        self.backend = backend

    def compute(
        self,
        *,
        pseudo_path: str | Path,
        reference_path: str | Path,
        target_shape: tuple[int, int],
        valid_mask: np.ndarray | None = None,
    ) -> Mast3RMatchResult:
        return self.backend.compute_confidence(
            pseudo_path=pseudo_path,
            reference_path=reference_path,
            target_shape=target_shape,
            valid_mask=valid_mask,
        )


def make_feature_backend(config: dict[str, Any]) -> Mast3RFeatureConfidence:
    pretrained = dict(config.get("pretrained", {}))
    feature = dict(pretrained.get("feature", {}))
    checkpoints = dict(pretrained.get("checkpoint_paths", {}))
    repos = dict(pretrained.get("repo_paths", {}))
    backend = Mast3RBackend(
        repo_root=repos.get("mast3r", ""),
        checkpoint_path=checkpoints.get("mast3r", ""),
        device=str(pretrained.get("device", "cuda")),
        image_size=int(feature.get("image_size", 512)),
        patch_size=int(feature.get("patch_size", 16)),
        min_confidence=float(feature.get("min_confidence", 0.05)),
        subsample=int(feature.get("subsample", 8)),
        block_size=int(feature.get("block_size", 8192)),
    )
    return Mast3RFeatureConfidence(backend)
