from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from part3_stack.coarse_geometry import (
    build_visibility_confidence,
    compute_anchor_reprojection,
    load_render_bundle,
    load_rgb_image,
    resize_rgb_to_shape,
)
from part3_stack.confidence import (
    CONFIDENCE_FLOOR,
    SOFT_CONFIDENCE_WEIGHTS,
    _detect_padding_validity,
    _map_to_preview,
    _save_component,
    build_patch_pruning_mask,
)
from part3_stack.config import ProjectConfig, ensure_workspace_dirs, load_config, read_json, write_json
from part3_stack.pipeline import build_hybrid, generate_pseudo_views, prepare_scene

from .clip_consistency import compute_clip_metrics
from .feature_confidence import make_feature_backend
from .temporal_confidence import make_temporal_backend
from .utils_pretrained import resize_map, save_rgb_array, stable_key


def _defaults_with_paths(config: ProjectConfig) -> dict[str, Any]:
    defaults = dict(config.defaults)
    pretrained = dict(defaults.get("pretrained", {}))

    def resolve_project_path(raw: str | Path) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = config.project_root / path
        return path.resolve()

    model_root = resolve_project_path(pretrained.get("model_root", config.part3_root))
    repo_paths = dict(pretrained.get("repo_paths", {}))
    repo_paths.setdefault("mast3r", str(config.part3_root / "mast3r"))
    repo_paths.setdefault("sea_raft", str(config.part3_root / "SEA-RAFT"))
    repo_paths = {key: str(resolve_project_path(value)) for key, value in repo_paths.items()}
    checkpoints = dict(pretrained.get("checkpoint_paths", {}))
    checkpoints.setdefault("mast3r", str(model_root / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"))
    checkpoints.setdefault("sea_raft", str(model_root / "Tartan480x640-M.pth"))
    checkpoints = {key: str(resolve_project_path(value)) for key, value in checkpoints.items()}
    temporal = dict(pretrained.get("temporal", {}))
    temporal.setdefault("backend", "sea_raft")
    temporal.setdefault("cfg_path", str(Path(repo_paths["sea_raft"]) / "config" / "train" / "Tartan480x640-M.json"))
    temporal.setdefault("flow_sigma", defaults.get("confidence", {}).get("flow_sigma", 1.5))
    temporal.setdefault("uncertainty_sigma", 1.0)
    temporal.setdefault("use_uncertainty", True)
    feature = dict(pretrained.get("feature", {}))
    feature.setdefault("backend", "mast3r")
    feature.setdefault("image_size", 512)
    feature.setdefault("patch_size", defaults.get("consistency", {}).get("patch_size", 16))
    feature.setdefault("min_confidence", 0.05)
    feature.setdefault("subsample", 8)
    pretrained.update(
        {
            "model_root": str(model_root),
            "repo_paths": repo_paths,
            "checkpoint_paths": checkpoints,
            "temporal": temporal,
            "feature": feature,
            "device": pretrained.get("device", "cuda"),
            "cache": bool(pretrained.get("cache", True)),
        }
    )
    defaults["pretrained"] = pretrained
    return defaults


def build_confidence_pretrained(
    *,
    config_path: str | Path | None = None,
    pseudo_manifest_path: str | Path,
    border_margin_ratio: float | None = None,
    reproj_sigma: float | None = None,
    alpha_threshold: float | None = None,
    boundary_percentile: float | None = None,
    boundary_dilate: int | None = None,
    enable_clip_consistency: bool | None = None,
    enable_patch_pruning: bool | None = None,
    patch_size: int | None = None,
    patch_threshold: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    defaults = _defaults_with_paths(config)
    manifest_path = build_confidence_manifest_pretrained(
        config=config,
        pseudo_manifest_path=pseudo_manifest_path,
        defaults=defaults,
        border_margin_ratio=border_margin_ratio,
        reproj_sigma=reproj_sigma,
        alpha_threshold=alpha_threshold,
        boundary_percentile=boundary_percentile,
        boundary_dilate=boundary_dilate,
        enable_clip_consistency=enable_clip_consistency,
        enable_patch_pruning=enable_patch_pruning,
        patch_size=patch_size,
        patch_threshold=patch_threshold,
    )
    return {"confidence_manifest_path": str(manifest_path)}


def build_confidence_manifest_pretrained(
    *,
    config: ProjectConfig,
    pseudo_manifest_path: str | Path,
    defaults: dict[str, Any],
    border_margin_ratio: float | None,
    reproj_sigma: float | None,
    alpha_threshold: float | None,
    boundary_percentile: float | None,
    boundary_dilate: int | None,
    enable_clip_consistency: bool | None,
    enable_patch_pruning: bool | None,
    patch_size: int | None,
    patch_threshold: float | None,
) -> Path:
    pseudo_manifest = read_json(pseudo_manifest_path)
    trajectory_manifest = read_json(pseudo_manifest["trajectory_manifest_path"])
    confidence_defaults = dict(defaults.get("confidence", {}))
    consistency_defaults = dict(defaults.get("consistency", {}))
    pretrained_defaults = dict(defaults.get("pretrained", {}))
    temporal_defaults = dict(pretrained_defaults.get("temporal", {}))
    feature_defaults = dict(pretrained_defaults.get("feature", {}))
    if str(temporal_defaults.get("backend", "sea_raft")) != "sea_raft":
        raise ValueError("The pretrained route currently supports temporal.backend='sea_raft' only.")
    if str(feature_defaults.get("backend", "mast3r")) != "mast3r":
        raise ValueError("The pretrained route currently supports feature.backend='mast3r' only.")

    alpha_threshold_value = float(alpha_threshold if alpha_threshold is not None else confidence_defaults.get("alpha_threshold", 0.05))
    reproj_sigma_value = float(reproj_sigma if reproj_sigma is not None else confidence_defaults.get("reproj_sigma", 0.2))
    border_margin_value = float(border_margin_ratio if border_margin_ratio is not None else confidence_defaults.get("border_margin_ratio", 0.0))
    boundary_percentile_value = float(boundary_percentile if boundary_percentile is not None else confidence_defaults.get("boundary_percentile", 90.0))
    boundary_dilate_value = int(boundary_dilate if boundary_dilate is not None else confidence_defaults.get("boundary_dilate", 3))
    clip_consistency_enabled = bool(
        consistency_defaults.get("enable_clip_consistency", True) if enable_clip_consistency is None else enable_clip_consistency
    )
    patch_pruning_enabled = bool(
        consistency_defaults.get("enable_patch_pruning", True) if enable_patch_pruning is None else enable_patch_pruning
    )
    patch_size_value = int(patch_size or consistency_defaults.get("patch_size", 16))
    patch_threshold_value = float(patch_threshold if patch_threshold is not None else consistency_defaults.get("patch_threshold", 0.25))
    patch_low_weight = float(consistency_defaults.get("patch_low_weight", 0.15))
    patch_min_keep_ratio = float(consistency_defaults.get("patch_min_keep_ratio", 0.1))
    clip_weights = dict(consistency_defaults.get("clip_weights", {}))
    component_flags = {
        "use_c_vis": True,
        "use_c_reproj": True,
        "use_c_feat": True,
        "use_c_temp": True,
        "use_clip_consistency": clip_consistency_enabled,
        "use_patch_pruning": patch_pruning_enabled,
    }
    component_flags.update(dict(defaults.get("ablation") or {}))

    confidence_root = Path(pseudo_manifest["run_dir"]) / "pretrained_confidence"
    confidence_root.mkdir(parents=True, exist_ok=True)
    cache_root = confidence_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    temporal_backend = make_temporal_backend(defaults) if component_flags["use_c_temp"] else None
    feature_backend = make_feature_backend(defaults) if component_flags["use_c_feat"] else None
    pair_lookup = {pair["pair_id"]: pair for pair in trajectory_manifest["pairs"]}
    manifest_clips: list[dict[str, Any]] = []

    for clip in pseudo_manifest["clips"]:
        pair = pair_lookup[clip["pair_id"]]
        frames: list[np.ndarray] = []
        for item in clip["assigned_views"]:
            frame = load_rgb_image(item["frame_path"])
            target_shape = (int(item["height"]), int(item["width"]))
            frames.append(resize_rgb_to_shape(frame, target_shape))
        if temporal_backend is not None:
            temporal_result = temporal_backend.compute_clip(frames)
            flow_maps = temporal_result.maps
        else:
            flow_maps = [np.ones(frame.shape[:2], dtype=np.float32) for frame in frames]
            temporal_result = None
        clip_dir = confidence_root / clip["pair_id"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        ref_dir = cache_root / clip["pair_id"] / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        mask_records: list[dict[str, Any]] = []

        for idx, (view_record, frame, flow_map) in enumerate(zip(clip["assigned_views"], frames, flow_maps)):
            c_padding = _detect_padding_validity(frame)
            if all(key in view_record for key in ("coarse_depth_path", "coarse_alpha_path", "coarse_normal_path", "coarse_rgb_path")):
                coarse_bundle = load_render_bundle(view_record)
                frame = resize_rgb_to_shape(frame, coarse_bundle.rgb.shape[:2])
                if flow_map.shape != frame.shape[:2]:
                    flow_map = resize_map(flow_map, frame.shape[:2])
                if c_padding.shape != coarse_bundle.rgb.shape[:2]:
                    pad_img = Image.fromarray(np.clip(c_padding * 255.0, 0, 255).astype(np.uint8), mode="L")
                    c_padding = np.asarray(
                        pad_img.resize((coarse_bundle.rgb.shape[1], coarse_bundle.rgb.shape[0]), Image.NEAREST),
                        dtype=np.float32,
                    ) / 255.0
                c_padding = (c_padding > 0.5).astype(np.float32)
                hard_validity_base = (
                    (coarse_bundle.alpha > alpha_threshold_value)
                    & np.isfinite(coarse_bundle.depth)
                    & (coarse_bundle.depth > 1e-4)
                ).astype(np.float32)
                c_vis = build_visibility_confidence(
                    coarse_bundle.depth,
                    coarse_bundle.alpha,
                    alpha_threshold=alpha_threshold_value,
                    boundary_percentile=boundary_percentile_value,
                    boundary_dilate=boundary_dilate_value,
                )
                if border_margin_value > 0:
                    h, w = c_vis.shape
                    margin = max(1, int(round(min(h, w) * border_margin_value)))
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
                    c_reproj[valid_reproj] = np.exp(-reproj["error_map"][valid_reproj] / max(1e-3, reproj_sigma_value))
                if feature_backend is not None:
                    ref_key = stable_key(clip["pair_id"], view_record["output_name"], "blend_rgb")
                    ref_path = save_rgb_array(ref_dir / f"{ref_key}_blend_rgb.png", reproj["blend_rgb"])
                    feature_result = feature_backend.compute(
                        pseudo_path=view_record["frame_path"],
                        reference_path=ref_path,
                        target_shape=frame.shape[:2],
                        valid_mask=reproj["valid_mask"],
                    )
                    c_feat = feature_result.confidence
                else:
                    feature_result = None
                    c_feat = np.ones_like(c_reproj, dtype=np.float32)
            else:
                h, w = frame.shape[:2]
                c_vis = np.ones((h, w), dtype=np.float32)
                c_reproj = np.ones((h, w), dtype=np.float32)
                c_feat = np.ones((h, w), dtype=np.float32)
                hard_validity_base = np.ones((h, w), dtype=np.float32)
                reproj = {"mean_error": 1.0, "valid_mask": np.zeros((h, w), dtype=np.float32)}
                feature_result = None

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
                    patch_size=patch_size_value,
                    threshold=patch_threshold_value,
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
                    "mast3r_num_matches": int(0 if feature_result is None else feature_result.num_matches),
                    "mast3r_match_density": float(0.0 if feature_result is None else feature_result.match_density),
                    "feature_backend": "mast3r",
                    "temporal_backend": "sea_raft",
                }
            )

        clip_metrics = compute_clip_metrics(mask_records, clip["assigned_views"], clip_weights)
        if not component_flags["use_clip_consistency"]:
            clip_metrics["clip_score"] = 1.0
        for record in mask_records:
            record["clip_score"] = float(clip_metrics["clip_score"])
        manifest_clips.append(
            {
                "pair_id": clip["pair_id"],
                "num_kept_frames": int(len(clip["assigned_views"])),
                "num_filtered_frames": int(len(clip.get("filtered_views", []))),
                "clip_metrics": clip_metrics,
                "sea_raft_pair_metrics": [] if temporal_result is None else temporal_result.pair_metrics,
                "mask_records": mask_records,
            }
        )

    payload = {
        "route": "pretrained",
        "scene": pseudo_manifest["scene"],
        "run_id": pseudo_manifest["run_id"],
        "run_dir": pseudo_manifest["run_dir"],
        "pseudo_manifest_path": str(Path(pseudo_manifest_path).expanduser()),
        "confidence_root": str(confidence_root),
        "flow_sigma": float(temporal_defaults.get("flow_sigma", 1.5)),
        "reproj_sigma": reproj_sigma_value,
        "alpha_threshold": alpha_threshold_value,
        "boundary_percentile": boundary_percentile_value,
        "boundary_dilate": boundary_dilate_value,
        "feature": {
            "backend": "mast3r",
            "checkpoint_path": dict(pretrained_defaults.get("checkpoint_paths", {})).get("mast3r", ""),
            "image_size": int(feature_defaults.get("image_size", 512)),
            "patch_size": int(feature_defaults.get("patch_size", patch_size_value)),
        },
        "temporal": {
            "backend": "sea_raft",
            "checkpoint_path": dict(pretrained_defaults.get("checkpoint_paths", {})).get("sea_raft", ""),
            "cfg_path": temporal_defaults.get("cfg_path", ""),
            "use_uncertainty": bool(temporal_defaults.get("use_uncertainty", True)),
        },
        "consistency": {
            "enable_clip_consistency": clip_consistency_enabled,
            "enable_patch_pruning": patch_pruning_enabled,
            "patch_size": patch_size_value,
            "patch_threshold": patch_threshold_value,
            "patch_low_weight": patch_low_weight,
            "patch_min_keep_ratio": patch_min_keep_ratio,
            "clip_weights": clip_weights,
        },
        "confidence_formula": {
            "type": "hard_validity_times_soft_floor",
            "version": 3,
            "floor": float(CONFIDENCE_FLOOR),
            "soft_weights": SOFT_CONFIDENCE_WEIGHTS,
            "hard_validity": "padding_validity_times_alpha_depth_validity",
            "pretrained_c_feat": "mast3r_reciprocal_match_density",
            "pretrained_c_temp": "sea_raft_forward_backward_consistency",
        },
        "ablation": component_flags,
        "config_snapshot": defaults,
        "clips": manifest_clips,
    }
    return write_json(confidence_root / "confidence_manifest.json", payload)


def build_hybrid_pretrained(
    *,
    config_path: str | Path | None = None,
    trajectory_manifest_path: str | Path,
    pseudo_manifest_path: str | Path,
    confidence_manifest_path: str | Path,
    hybrid_name: str | None = None,
    train: bool = False,
    output_tag: str | None = None,
    iterations: int | None = None,
) -> dict[str, Any]:
    return build_hybrid(
        config_path=config_path,
        trajectory_manifest_path=trajectory_manifest_path,
        pseudo_manifest_path=pseudo_manifest_path,
        confidence_manifest_path=confidence_manifest_path,
        hybrid_name=hybrid_name,
        train=train,
        output_tag=output_tag,
        iterations=iterations,
    )


def run_part3_pretrained(
    *,
    config_path: str | Path | None = None,
    scene: str,
    pseudo_manifest_path: str | Path | None = None,
    trajectory_manifest_path: str | Path | None = None,
    prompt: str | None = None,
    num_intermediate_views: int | None = None,
    max_pairs: int | None = None,
    run_id: str | None = None,
    dynami_crafter: dict[str, Any] | None = None,
    keep_ratio: float | None = None,
    hybrid_name: str | None = None,
    train: bool = False,
    output_tag: str | None = None,
    iterations: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if pseudo_manifest_path:
        pseudo_manifest = read_json(pseudo_manifest_path)
        resolved_trajectory_manifest = str(trajectory_manifest_path or pseudo_manifest["trajectory_manifest_path"])
        pseudo_result = {"pseudo_manifest_path": str(Path(pseudo_manifest_path).expanduser())}
        scene_result = {"trajectory_manifest_path": resolved_trajectory_manifest, "scene": scene}
    else:
        scene_result = prepare_scene(
            config_path=config.config_path,
            scene=scene,
            num_intermediate_views=num_intermediate_views,
            max_pairs=max_pairs,
            run_id=run_id,
        )
        pseudo_result = generate_pseudo_views(
            config_path=config.config_path,
            trajectory_manifest_path=scene_result["trajectory_manifest_path"],
            prompt=prompt,
            dynami_crafter=dynami_crafter,
            keep_ratio=keep_ratio,
        )
        resolved_trajectory_manifest = scene_result["trajectory_manifest_path"]

    confidence_result = build_confidence_pretrained(
        config_path=config.config_path,
        pseudo_manifest_path=pseudo_result["pseudo_manifest_path"],
    )
    hybrid_result = build_hybrid_pretrained(
        config_path=config.config_path,
        trajectory_manifest_path=resolved_trajectory_manifest,
        pseudo_manifest_path=pseudo_result["pseudo_manifest_path"],
        confidence_manifest_path=confidence_result["confidence_manifest_path"],
        hybrid_name=hybrid_name,
        train=train,
        output_tag=output_tag,
        iterations=iterations,
    )
    return {
        "route": "pretrained",
        "scene": scene_result,
        "pseudo_view": pseudo_result,
        "confidence": confidence_result,
        "hybrid": hybrid_result,
    }
