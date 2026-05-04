from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .coarse_geometry import (
    build_visibility_confidence,
    compute_anchor_reprojection,
    load_render_bundle,
    load_rgb_image,
    resize_rgb_to_shape,
)
from .config import ProjectConfig, read_json, write_json


CONFIDENCE_FLOOR = 0.3
SOFT_CONFIDENCE_WEIGHTS = {
    "reproj": 0.4,
    "feat": 0.3,
    "temp": 0.3,
}


class FeatureBackend:
    """Small dense feature interface used by C_feat.

    The default backend is named "dust3r" so the manifest and CLI line up with
    the planned DUSt3R path. Its first implementation is a light dense
    descriptor that keeps the pipeline fast and dependency-safe; replacing
    `extract` with true DUSt3R tokens later will not change downstream code.
    """

    def __init__(self, name: str, stride: int = 2, cache: bool = True) -> None:
        self.name = name
        self.stride = max(1, int(stride))
        self.cache_enabled = bool(cache)
        self._cache: dict[str, np.ndarray] = {}

    def extract(self, image: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        if cache_key and self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        cv2 = _require_cv2()
        rgb = image.astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        if self.stride > 1:
            small_w = max(8, int(np.ceil(w / self.stride)))
            small_h = max(8, int(np.ceil(h / self.stride)))
            work = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            work = rgb

        gray = cv2.cvtColor(np.clip(work * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        descriptor = np.concatenate(
            [
                work,
                gray[..., None],
                sobel_x[..., None],
                sobel_y[..., None],
                lap[..., None],
            ],
            axis=2,
        ).astype(np.float32)
        descriptor -= descriptor.mean(axis=(0, 1), keepdims=True)
        descriptor /= descriptor.std(axis=(0, 1), keepdims=True) + 1e-6
        norm = np.linalg.norm(descriptor, axis=2, keepdims=True)
        descriptor = descriptor / np.maximum(norm, 1e-6)
        if descriptor.shape[:2] != (h, w):
            descriptor = cv2.resize(descriptor, (w, h), interpolation=cv2.INTER_LINEAR)
            descriptor /= np.maximum(np.linalg.norm(descriptor, axis=2, keepdims=True), 1e-6)

        if cache_key and self.cache_enabled:
            self._cache[cache_key] = descriptor
        return descriptor


def make_feature_backend(name: str, stride: int = 2, cache: bool = True) -> FeatureBackend:
    return FeatureBackend(name=str(name or "dust3r"), stride=stride, cache=cache)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for confidence-map construction. "
            "Install opencv-python in the environment used for Part 3."
        ) from exc
    return cv2


def _gray(frame: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _pair_flow_confidence(first: np.ndarray, second: np.ndarray, sigma: float) -> np.ndarray:
    cv2 = _require_cv2()
    gray_a = _gray(first)
    gray_b = _gray(second)
    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = gray_a.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_ab[..., 0]
    map_y = grid_y + flow_ab[..., 1]
    warped_ba = cv2.remap(flow_ba, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    fb_error = np.linalg.norm(flow_ab + warped_ba, axis=2)
    sigma = max(1e-3, float(sigma))
    return np.exp(-fb_error / sigma).astype(np.float32)


def _aggregate_flow_confidences(frames: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    if not frames:
        return []
    if len(frames) == 1:
        h, w = frames[0].shape[:2]
        return [np.ones((h, w), dtype=np.float32)]

    pair_maps = [_pair_flow_confidence(frames[i], frames[i + 1], sigma) for i in range(len(frames) - 1)]
    outputs: list[np.ndarray] = []
    for idx in range(len(frames)):
        if idx == 0:
            outputs.append(pair_maps[0])
        elif idx == len(frames) - 1:
            outputs.append(pair_maps[-1])
        else:
            outputs.append(0.5 * (pair_maps[idx - 1] + pair_maps[idx]))
    return outputs


def _map_to_preview(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")


def _save_component(prefix: Path, suffix: str, array: np.ndarray) -> tuple[str, str]:
    npy_path = prefix.with_name(f"{prefix.name}_{suffix}.npy")
    preview_path = prefix.with_name(f"{prefix.name}_{suffix}.png")
    np.save(npy_path, array.astype(np.float32))
    _map_to_preview(array).save(preview_path)
    return str(npy_path), str(preview_path)


def _detect_padding_validity(
    rgb: np.ndarray,
    *,
    min_band_px: int = 4,
    color_threshold: float = 18.0,
    min_fraction: float = 0.92,
    max_padding_fraction: float = 0.25,
) -> np.ndarray:
    """Return 0 only for near-constant padding bands connected to image edges.

    This is intentionally conservative: true DynamiCrafter letterbox/pad bands
    are near-uniform and touch the border, while real image content near the
    border should keep its confidence floor.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.ones(rgb.shape[:2], dtype=np.float32)

    h, w = rgb.shape[:2]
    if h < 2 * min_band_px or w < 2 * min_band_px:
        return np.ones((h, w), dtype=np.float32)

    arr = rgb[..., :3].astype(np.float32)
    corner = max(2, min(8, min(h, w) // 16))
    corner_pixels = np.concatenate(
        [
            arr[:corner, :corner].reshape(-1, 3),
            arr[:corner, -corner:].reshape(-1, 3),
            arr[-corner:, :corner].reshape(-1, 3),
            arr[-corner:, -corner:].reshape(-1, 3),
        ],
        axis=0,
    )
    pad_color = np.median(corner_pixels, axis=0)
    color_dist = np.linalg.norm(arr - pad_color[None, None, :], axis=2)
    padding_like = color_dist <= float(color_threshold)
    valid = np.ones((h, w), dtype=np.float32)

    def scan_vertical(from_left: bool) -> int:
        limit = max(min_band_px, int(round(w * max_padding_fraction)))
        width = 0
        for offset in range(limit):
            x = offset if from_left else w - 1 - offset
            if float(padding_like[:, x].mean()) < min_fraction:
                break
            width += 1
        return width if width >= min_band_px else 0

    def scan_horizontal(from_top: bool) -> int:
        limit = max(min_band_px, int(round(h * max_padding_fraction)))
        height = 0
        for offset in range(limit):
            y = offset if from_top else h - 1 - offset
            if float(padding_like[y, :].mean()) < min_fraction:
                break
            height += 1
        return height if height >= min_band_px else 0

    left = scan_vertical(True)
    right = scan_vertical(False)
    top = scan_horizontal(True)
    bottom = scan_horizontal(False)
    if left:
        valid[:, :left] = 0.0
    if right:
        valid[:, w - right :] = 0.0
    if top:
        valid[:top, :] = 0.0
    if bottom:
        valid[h - bottom :, :] = 0.0
    return valid


def _feature_confidence(
    *,
    backend: FeatureBackend,
    pseudo_rgb: np.ndarray,
    warped_anchor_rgb: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
    pseudo_key: str,
    anchor_key: str,
) -> np.ndarray:
    pseudo_feat = backend.extract(pseudo_rgb, cache_key=pseudo_key)
    anchor_feat = backend.extract(warped_anchor_rgb, cache_key=anchor_key)
    similarity = np.sum(pseudo_feat * anchor_feat, axis=2)
    similarity = np.clip((similarity + 1.0) * 0.5, 0.0, 1.0)
    sigma = max(1e-3, float(sigma))
    confidence = np.exp(-(1.0 - similarity) / sigma).astype(np.float32)
    confidence[valid_mask <= 0] = 0.0
    return confidence


def build_patch_pruning_mask(
    confidence: np.ndarray,
    *,
    patch_size: int,
    threshold: float,
    low_weight: float,
    min_keep_ratio: float,
) -> tuple[np.ndarray, float]:
    h, w = confidence.shape
    patch = max(1, int(patch_size))
    threshold = float(threshold)
    low_weight = float(np.clip(low_weight, 0.0, 1.0))
    min_keep_ratio = float(np.clip(min_keep_ratio, 0.0, 1.0))

    padded_h = int(np.ceil(h / patch) * patch)
    padded_w = int(np.ceil(w / patch) * patch)
    padded = np.pad(confidence, ((0, padded_h - h), (0, padded_w - w)), mode="edge")
    grid = padded.reshape(padded_h // patch, patch, padded_w // patch, patch).mean(axis=(1, 3))
    keep = grid >= threshold
    if keep.size and keep.mean() < min_keep_ratio:
        keep_count = max(1, int(np.ceil(keep.size * min_keep_ratio)))
        flat_order = np.argsort(grid.reshape(-1))[::-1]
        keep = np.zeros_like(grid, dtype=bool)
        flat_keep = keep.reshape(-1)
        flat_keep[flat_order[:keep_count]] = True
        keep = flat_keep.reshape(grid.shape)

    patch_mask = np.repeat(np.repeat(keep.astype(np.float32), patch, axis=0), patch, axis=1)[:h, :w]
    patch_mask = low_weight + (1.0 - low_weight) * patch_mask
    return patch_mask.astype(np.float32), float(np.mean(patch_mask > low_weight + 1e-6))


def _qvec_to_rotmat(qvec: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q = q / max(1e-12, np.linalg.norm(q))
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_angle(rot: np.ndarray) -> float:
    trace = float(np.trace(rot))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _pose_smoothness(records: list[dict[str, Any]]) -> dict[str, float]:
    if len(records) < 3:
        return {"score": 1.0, "translation_std": 0.0, "rotation_std": 0.0}

    ordered = sorted(records, key=lambda item: float(item.get("alpha", item.get("index", 0))))
    tvecs = [np.asarray(item["tvec"], dtype=np.float64) for item in ordered]
    rots = [_qvec_to_rotmat(item["qvec"]) for item in ordered]
    trans_steps = np.asarray([np.linalg.norm(tvecs[i + 1] - tvecs[i]) for i in range(len(tvecs) - 1)], dtype=np.float64)
    rot_steps = np.asarray([_rotation_angle(rots[i + 1] @ rots[i].T) for i in range(len(rots) - 1)], dtype=np.float64)
    trans_std = float(np.std(trans_steps)) if trans_steps.size else 0.0
    rot_std = float(np.std(rot_steps)) if rot_steps.size else 0.0
    score = float(np.exp(-(trans_std + rot_std)))
    return {"score": score, "translation_std": trans_std, "rotation_std": rot_std}


def _weighted_clip_score(records: list[dict[str, Any]], pose_score: float, weights: dict[str, float]) -> dict[str, float]:
    if records:
        s_flow = float(np.mean([record.get("mean_temporal_confidence", 0.0) for record in records]))
        s_reproj = float(np.mean([record.get("mean_reprojection_confidence", 0.0) for record in records]))
        s_feat = float(np.mean([record.get("mean_feature_confidence", 0.0) for record in records]))
    else:
        s_flow = s_reproj = s_feat = 0.0
    w_flow = float(weights.get("flow", 0.25))
    w_reproj = float(weights.get("reproj", 0.35))
    w_feat = float(weights.get("feat", 0.25))
    w_pose = float(weights.get("pose", 0.15))
    denom = max(1e-6, w_flow + w_reproj + w_feat + w_pose)
    score = (w_flow * s_flow + w_reproj * s_reproj + w_feat * s_feat + w_pose * pose_score) / denom
    return {
        "clip_score": float(np.clip(score, 0.0, 1.0)),
        "s_flow": s_flow,
        "s_reproj": s_reproj,
        "s_feat": s_feat,
        "s_pose": float(np.clip(pose_score, 0.0, 1.0)),
    }


def build_confidence_manifest(
    config: ProjectConfig,
    pseudo_manifest_path: str | Path,
    border_margin_ratio: float,
    flow_sigma: float,
    anchor_sigma: float,
    reproj_sigma: float,
    alpha_threshold: float,
    boundary_percentile: float,
    boundary_dilate: int,
    feature_backend: str,
    feature_sigma: float,
    feature_stride: int,
    feature_cache: bool,
    enable_clip_consistency: bool,
    enable_patch_pruning: bool,
    patch_size: int,
    patch_threshold: float,
    patch_low_weight: float,
    patch_min_keep_ratio: float,
    ablation: dict[str, bool] | None = None,
) -> Path:
    pseudo_manifest = read_json(pseudo_manifest_path)
    trajectory_manifest = read_json(pseudo_manifest["trajectory_manifest_path"])
    confidence_root = Path(pseudo_manifest["run_dir"]) / "confidence"
    confidence_root.mkdir(parents=True, exist_ok=True)

    pair_lookup = {pair["pair_id"]: pair for pair in trajectory_manifest["pairs"]}
    manifest_clips: list[dict[str, Any]] = []
    backend = make_feature_backend(feature_backend, stride=feature_stride, cache=feature_cache)
    component_flags = {
        "use_c_vis": True,
        "use_c_reproj": True,
        "use_c_feat": True,
        "use_c_temp": True,
        "use_clip_consistency": bool(enable_clip_consistency),
        "use_patch_pruning": bool(enable_patch_pruning),
    }
    component_flags.update(ablation or {})
    consistency_defaults = dict(config.defaults.get("consistency", {}))
    clip_weights = dict(consistency_defaults.get("clip_weights", {}))

    for clip in pseudo_manifest["clips"]:
        pair = pair_lookup[clip["pair_id"]]
        frames = []
        for item in clip["assigned_views"]:
            frame = load_rgb_image(item["frame_path"])
            target_shape = (int(item["height"]), int(item["width"]))
            frames.append(resize_rgb_to_shape(frame, target_shape))
        flow_maps = _aggregate_flow_confidences(frames, flow_sigma)

        clip_dir = confidence_root / clip["pair_id"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        mask_records: list[dict[str, Any]] = []

        for idx, (view_record, frame, flow_map) in enumerate(zip(clip["assigned_views"], frames, flow_maps)):
            hard_validity_base: np.ndarray | None = None
            c_padding = _detect_padding_validity(frame)
            if all(key in view_record for key in ("coarse_depth_path", "coarse_alpha_path", "coarse_normal_path", "coarse_rgb_path")):
                coarse_bundle = load_render_bundle(view_record)
                frame = resize_rgb_to_shape(frame, coarse_bundle.rgb.shape[:2])
                if c_padding.shape != coarse_bundle.rgb.shape[:2]:
                    pad_img = Image.fromarray(np.clip(c_padding * 255.0, 0, 255).astype(np.uint8), mode="L")
                    c_padding = np.asarray(
                        pad_img.resize((coarse_bundle.rgb.shape[1], coarse_bundle.rgb.shape[0]), Image.NEAREST),
                        dtype=np.float32,
                    ) / 255.0
                c_padding = (c_padding > 0.5).astype(np.float32)
                hard_validity_base = (
                    (coarse_bundle.alpha > float(alpha_threshold))
                    & np.isfinite(coarse_bundle.depth)
                    & (coarse_bundle.depth > 1e-4)
                ).astype(np.float32)
                c_vis = build_visibility_confidence(
                    coarse_bundle.depth,
                    coarse_bundle.alpha,
                    alpha_threshold=alpha_threshold,
                    boundary_percentile=boundary_percentile,
                    boundary_dilate=boundary_dilate,
                )
                if border_margin_ratio > 0:
                    h, w = c_vis.shape
                    margin = max(1, int(round(min(h, w) * float(border_margin_ratio))))
                    border_mask = np.zeros_like(c_vis, dtype=np.float32)
                    if h <= 2 * margin or w <= 2 * margin:
                        border_mask.fill(1.0)
                    else:
                        border_mask[margin:-margin, margin:-margin] = 1.0
                    c_vis *= border_mask
                reproj = compute_anchor_reprojection(
                    project_root=config.project_root,
                    pseudo_rgb=frame,
                    target_qvec=view_record["qvec"],
                    target_tvec=view_record["tvec"],
                    target_camera_model=str(view_record["camera_model"]),
                    target_camera_params=view_record["camera_params"],
                    target_depth=coarse_bundle.depth,
                    start_anchor_path=pair["start_image_path"],
                    start_qvec=pair["start_qvec"],
                    start_tvec=pair["start_tvec"],
                    start_camera_model=str(pair.get("start_camera_model", pair["camera_model"])),
                    start_camera_params=pair.get("start_camera_params", pair["camera_params"]),
                    end_anchor_path=pair["end_image_path"],
                    end_qvec=pair["end_qvec"],
                    end_tvec=pair["end_tvec"],
                    end_camera_model=str(pair.get("end_camera_model", pair["camera_model"])),
                    end_camera_params=pair.get("end_camera_params", pair["camera_params"]),
                    alpha=float(view_record["alpha"]),
                )
                c_reproj = np.zeros_like(flow_map, dtype=np.float32)
                valid_reproj = reproj["valid_mask"] > 0
                if np.any(valid_reproj):
                    c_reproj[valid_reproj] = np.exp(-reproj["error_map"][valid_reproj] / max(1e-3, float(reproj_sigma)))
                c_feat = _feature_confidence(
                    backend=backend,
                    pseudo_rgb=frame,
                    warped_anchor_rgb=reproj["blend_rgb"],
                    valid_mask=reproj["valid_mask"],
                    sigma=feature_sigma,
                    pseudo_key=str(view_record["frame_path"]),
                    anchor_key=f"{clip['pair_id']}:{view_record['output_name']}:warped_anchor",
                )
            else:
                h, w = frame.shape[:2]
                c_vis = np.ones((h, w), dtype=np.float32)
                c_reproj = np.ones((h, w), dtype=np.float32)
                c_feat = np.ones((h, w), dtype=np.float32)
                hard_validity_base = np.ones((h, w), dtype=np.float32)
                reproj = {"mean_error": 1.0, "valid_mask": np.zeros((h, w), dtype=np.float32)}

            c_reproj_used = c_reproj if component_flags["use_c_reproj"] else np.ones_like(c_reproj, dtype=np.float32)
            c_feat_used = c_feat if component_flags["use_c_feat"] else np.ones_like(c_feat, dtype=np.float32)
            c_temp_used = flow_map if component_flags["use_c_temp"] else np.ones_like(flow_map, dtype=np.float32)
            soft_confidence = np.clip(
                SOFT_CONFIDENCE_WEIGHTS["reproj"] * c_reproj_used
                + SOFT_CONFIDENCE_WEIGHTS["feat"] * c_feat_used
                + SOFT_CONFIDENCE_WEIGHTS["temp"] * c_temp_used,
                0.0,
                1.0,
            ).astype(np.float32)
            hard_validity = (
                np.clip(hard_validity_base, 0.0, 1.0).astype(np.float32)
                if component_flags["use_c_vis"]
                else np.ones_like(c_vis, dtype=np.float32)
            )
            hard_validity = np.clip(hard_validity * c_padding, 0.0, 1.0).astype(np.float32)
            pre_patch_confidence = np.clip(
                hard_validity * (CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * soft_confidence),
                0.0,
                1.0,
            ).astype(np.float32)
            if component_flags["use_patch_pruning"]:
                c_patch, patch_keep_ratio = build_patch_pruning_mask(
                    pre_patch_confidence,
                    patch_size=patch_size,
                    threshold=patch_threshold,
                    low_weight=patch_low_weight,
                    min_keep_ratio=patch_min_keep_ratio,
                )
            else:
                c_patch = np.ones_like(pre_patch_confidence, dtype=np.float32)
                patch_keep_ratio = 1.0
            fused = np.clip(
                hard_validity * c_patch * (CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * soft_confidence),
                0.0,
                1.0,
            ).astype(np.float32)

            prefix = clip_dir / f"{clip['pair_id']}_{idx:02d}"
            mask_path = prefix.with_name(f"{prefix.name}_confidence.npy")
            preview_path = prefix.with_name(f"{prefix.name}_confidence.png")
            raw_confidence_path, raw_confidence_preview_path = _save_component(prefix, "confidence_raw", pre_patch_confidence)
            soft_confidence_path, soft_confidence_preview_path = _save_component(prefix, "confidence_soft", soft_confidence)
            hard_validity_path, hard_validity_preview_path = _save_component(prefix, "hard_validity", hard_validity)
            c_padding_path, c_padding_preview_path = _save_component(prefix, "c_padding", c_padding)
            c_vis_path, c_vis_preview_path = _save_component(prefix, "c_vis", c_vis)
            c_reproj_path, c_reproj_preview_path = _save_component(prefix, "c_reproj", c_reproj)
            c_feat_path, c_feat_preview_path = _save_component(prefix, "c_feat", c_feat)
            c_temp_path, c_temp_preview_path = _save_component(prefix, "c_temp", flow_map)
            c_patch_path, c_patch_preview_path = _save_component(prefix, "c_patch", c_patch)
            np.save(mask_path, fused)
            _map_to_preview(fused).save(preview_path)

            mask_records.append(
                {
                    "index": idx,
                    "frame_path": view_record["frame_path"],
                    "mask_path": str(mask_path),
                    "preview_path": str(preview_path),
                    "raw_confidence_path": raw_confidence_path,
                    "raw_confidence_preview_path": raw_confidence_preview_path,
                    "soft_confidence_path": soft_confidence_path,
                    "soft_confidence_preview_path": str(soft_confidence_preview_path),
                    "hard_validity_path": hard_validity_path,
                    "hard_validity_preview_path": str(hard_validity_preview_path),
                    "c_padding_path": c_padding_path,
                    "c_padding_preview_path": str(c_padding_preview_path),
                    "c_vis_path": c_vis_path,
                    "c_vis_preview_path": str(c_vis_preview_path),
                    "c_reproj_path": c_reproj_path,
                    "c_reproj_preview_path": str(c_reproj_preview_path),
                    "c_feat_path": c_feat_path,
                    "c_feat_preview_path": str(c_feat_preview_path),
                    "c_temp_path": c_temp_path,
                    "c_temp_preview_path": str(c_temp_preview_path),
                    "c_patch_path": c_patch_path,
                    "c_patch_preview_path": str(c_patch_preview_path),
                    "coarse_depth_path": view_record.get("coarse_depth_path", ""),
                    "coarse_rgb_path": view_record.get("coarse_rgb_path", ""),
                    "mean_confidence": float(fused.mean()),
                    "mean_visibility_confidence": float(c_vis.mean()),
                    "mean_reprojection_confidence": float(c_reproj.mean()),
                    "mean_feature_confidence": float(c_feat.mean()),
                    "mean_temporal_confidence": float(flow_map.mean()),
                    "mean_raw_confidence": float(pre_patch_confidence.mean()),
                    "mean_soft_confidence": float(soft_confidence.mean()),
                    "mean_hard_validity": float(hard_validity.mean()),
                    "mean_padding_validity": float(c_padding.mean()),
                    "mean_patch_weight": float(c_patch.mean()),
                    "mean_final_mask": float(fused.mean()),
                    "patch_keep_ratio": float(patch_keep_ratio),
                    "mean_reprojection_error": float(reproj["mean_error"]),
                    "reprojection_valid_ratio": float(np.mean(reproj["valid_mask"] > 0)),
                    "alpha": float(view_record["alpha"]),
                    "output_name": view_record["output_name"],
                    "frame_score": float(view_record.get("frame_score", 0.0)),
                    "pose_refined": bool(view_record.get("pose_refined", False)),
                }
            )

        pose_smoothness = _pose_smoothness(clip["assigned_views"])
        clip_metrics = _weighted_clip_score(
            mask_records,
            pose_score=pose_smoothness["score"],
            weights=clip_weights,
        )
        if not component_flags["use_clip_consistency"]:
            clip_metrics["clip_score"] = 1.0
        for record in mask_records:
            record["clip_score"] = float(clip_metrics["clip_score"])

        manifest_clips.append(
            {
                "pair_id": clip["pair_id"],
                "num_kept_frames": int(len(clip["assigned_views"])),
                "num_filtered_frames": int(len(clip.get("filtered_views", []))),
                "clip_metrics": {
                    **clip_metrics,
                    "pose_translation_std": pose_smoothness["translation_std"],
                    "pose_rotation_std": pose_smoothness["rotation_std"],
                },
                "mask_records": mask_records,
            }
        )

    payload = {
        "scene": pseudo_manifest["scene"],
        "run_id": pseudo_manifest["run_id"],
        "run_dir": pseudo_manifest["run_dir"],
        "pseudo_manifest_path": str(Path(pseudo_manifest_path).expanduser()),
        "confidence_root": str(confidence_root),
        "flow_sigma": float(flow_sigma),
        "reproj_sigma": float(reproj_sigma),
        "anchor_sigma": float(anchor_sigma),
        "alpha_threshold": float(alpha_threshold),
        "boundary_percentile": float(boundary_percentile),
        "boundary_dilate": int(boundary_dilate),
        "feature": {
            "backend": backend.name,
            "sigma": float(feature_sigma),
            "stride": int(feature_stride),
            "cache": bool(feature_cache),
        },
        "consistency": {
            "enable_clip_consistency": bool(enable_clip_consistency),
            "enable_patch_pruning": bool(enable_patch_pruning),
            "patch_size": int(patch_size),
            "patch_threshold": float(patch_threshold),
            "patch_low_weight": float(patch_low_weight),
            "patch_min_keep_ratio": float(patch_min_keep_ratio),
            "clip_weights": clip_weights,
        },
        "confidence_formula": {
            "type": "hard_validity_times_soft_floor",
            "version": 3,
            "floor": float(CONFIDENCE_FLOOR),
            "soft_weights": SOFT_CONFIDENCE_WEIGHTS,
            "hard_validity": "padding_validity_times_alpha_depth_validity",
            "padding": {
                "type": "edge_connected_near_constant_band",
            },
            "patch": {
                "type": "soft_weight",
                "low_weight": float(patch_low_weight),
            },
        },
        "ablation": component_flags,
        "config_snapshot": config.defaults,
        "clips": manifest_clips,
    }
    return write_json(confidence_root / "confidence_manifest.json", payload)
