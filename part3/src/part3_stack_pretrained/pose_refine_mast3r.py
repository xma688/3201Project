from __future__ import annotations

from typing import Any


def maybe_refine_pose_with_mast3r(*, enabled: bool, **_kwargs: Any) -> dict[str, Any]:
    """Reserved hook for a future MASt3R PnP fallback.

    The pretrained route currently keeps pose refinement on the existing ORB+PnP
    path during pseudo generation and only uses MASt3R for offline feature
    confidence. Returning a structured disabled result keeps the pipeline
    explicit without changing pose behavior.
    """

    return {
        "enabled": bool(enabled),
        "success": False,
        "reason": "MASt3R pose fallback is reserved but not enabled in the v1 pretrained route.",
    }
