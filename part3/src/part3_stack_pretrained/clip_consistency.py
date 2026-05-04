from __future__ import annotations

from typing import Any

from part3_stack.confidence import _pose_smoothness, _weighted_clip_score


def compute_clip_metrics(
    records: list[dict[str, Any]],
    view_records: list[dict[str, Any]],
    clip_weights: dict[str, float],
) -> dict[str, float]:
    pose = _pose_smoothness(view_records)
    metrics = _weighted_clip_score(records, pose_score=float(pose["score"]), weights=clip_weights)
    return {
        **metrics,
        "pose_translation_std": float(pose["translation_std"]),
        "pose_rotation_std": float(pose["rotation_std"]),
    }
