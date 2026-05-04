from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sea_raft_backend import SeaRaftBackend


@dataclass(frozen=True)
class TemporalClipResult:
    maps: list[np.ndarray]
    pair_metrics: list[dict[str, float]]


class SeaRaftTemporalConfidence:
    def __init__(
        self,
        backend: SeaRaftBackend,
        *,
        sigma: float,
        uncertainty_sigma: float,
    ) -> None:
        self.backend = backend
        self.sigma = float(sigma)
        self.uncertainty_sigma = float(uncertainty_sigma)

    def compute_clip(self, frames: list[np.ndarray]) -> TemporalClipResult:
        if not frames:
            return TemporalClipResult(maps=[], pair_metrics=[])
        if len(frames) == 1:
            h, w = frames[0].shape[:2]
            return TemporalClipResult(maps=[np.ones((h, w), dtype=np.float32)], pair_metrics=[])

        pair_maps: list[np.ndarray] = []
        pair_metrics: list[dict[str, float]] = []
        for idx in range(len(frames) - 1):
            conf, metrics = self.backend.pair_confidence(
                frames[idx],
                frames[idx + 1],
                sigma=self.sigma,
                uncertainty_sigma=self.uncertainty_sigma,
            )
            pair_maps.append(conf)
            pair_metrics.append({"pair_index": float(idx), **metrics})

        outputs: list[np.ndarray] = []
        for idx in range(len(frames)):
            if idx == 0:
                outputs.append(pair_maps[0])
            elif idx == len(frames) - 1:
                outputs.append(pair_maps[-1])
            else:
                outputs.append(np.clip(0.5 * (pair_maps[idx - 1] + pair_maps[idx]), 0.0, 1.0).astype(np.float32))
        return TemporalClipResult(maps=outputs, pair_metrics=pair_metrics)


def make_temporal_backend(config: dict[str, Any]) -> SeaRaftTemporalConfidence:
    pretrained = dict(config.get("pretrained", {}))
    temporal = dict(pretrained.get("temporal", {}))
    checkpoints = dict(pretrained.get("checkpoint_paths", {}))
    repos = dict(pretrained.get("repo_paths", {}))
    backend = SeaRaftBackend(
        repo_root=repos.get("sea_raft", ""),
        checkpoint_path=checkpoints.get("sea_raft", ""),
        cfg_path=temporal.get("cfg_path", ""),
        device=str(pretrained.get("device", "cuda")),
        iters=temporal.get("iters"),
        use_uncertainty=bool(temporal.get("use_uncertainty", True)),
    )
    return SeaRaftTemporalConfidence(
        backend,
        sigma=float(temporal.get("flow_sigma", 1.5)),
        uncertainty_sigma=float(temporal.get("uncertainty_sigma", 1.0)),
    )
