from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute() or base is None:
        return raw
    return (base / raw).expanduser()


def stable_key(*parts: object, length: int = 16) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def save_rgb_array(path: str | Path, rgb: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(out)
    return out


def resize_map(array: np.ndarray, shape_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape[:2] == (h, w):
        return arr.astype(np.float32, copy=False)
    mode = Image.NEAREST if nearest else Image.BILINEAR
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((w, h), mode), dtype=np.float32) / 255.0


def confidence_from_patch_values(
    shape_hw: tuple[int, int],
    points_xy: np.ndarray,
    *,
    patch_size: int,
    min_value: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    patch = max(1, int(patch_size))
    grid_h = int(np.ceil(h / patch))
    grid_w = int(np.ceil(w / patch))
    density = np.zeros((grid_h, grid_w), dtype=np.float32)
    if points_xy.size:
        xy = np.asarray(points_xy, dtype=np.float32)
        xs = np.clip(np.floor(xy[:, 0] / patch).astype(np.int64), 0, grid_w - 1)
        ys = np.clip(np.floor(xy[:, 1] / patch).astype(np.int64), 0, grid_h - 1)
        np.add.at(density, (ys, xs), 1.0)
    if density.max() > 0:
        scale = float(np.percentile(density[density > 0], 90.0))
        density = np.clip(density / max(1.0, scale), 0.0, 1.0)
    conf_small = min_value + (1.0 - min_value) * density
    conf = np.repeat(np.repeat(conf_small, patch, axis=0), patch, axis=1)[:h, :w]
    if valid_mask is not None:
        conf = conf * (np.asarray(valid_mask, dtype=np.float32) > 0).astype(np.float32)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def get_nested(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
