from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .coarse_geometry import (
    CoarseGeometryRenderer,
    compute_anchor_reprojection,
    load_render_bundle,
    load_rgb_image,
    refine_pose_with_pnp,
    render_trajectory_coarse_guidance,
    resize_rgb_to_shape,
    save_render_bundle,
    save_rgb_image,
    score_pseudo_frame,
)
from .confidence import build_confidence_manifest
from .config import ensure_workspace_dirs, load_config, read_json, write_json
from .geometry import (
    build_trajectory_manifest,
    resolve_base_scene,
    suggest_scene_build_commands,
)
from .hybrid import build_hybrid_scene, launch_evaluation, launch_training


_ADAPTERS: dict[str, Any] = {}


def _get_dynami_modules():
    from .dynamicrafter_adapter import DynamiCrafterInterpolationAdapter, extract_uniform_frames

    return DynamiCrafterInterpolationAdapter, extract_uniform_frames


def _get_adapter(config):
    DynamiCrafterInterpolationAdapter, _extract_uniform_frames = _get_dynami_modules()
    key = str(config.config_path)
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        adapter = DynamiCrafterInterpolationAdapter(config)
        _ADAPTERS[key] = adapter
    return adapter


def _split_kept_views(view_records: list[dict[str, Any]], keep_ratio: float, min_keep_per_clip: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not view_records:
        return [], []
    clamped_ratio = min(1.0, max(0.0, float(keep_ratio)))
    keep_count = len(view_records) if clamped_ratio >= 0.999 else max(min_keep_per_clip, int(round(len(view_records) * clamped_ratio)))
    keep_count = max(1, min(len(view_records), keep_count))
    selected_names = {
        item["output_name"]
        for item in sorted(view_records, key=lambda record: float(record.get("frame_score", 1.0)))[:keep_count]
    }
    kept = [{**record, "filtered_out": False} for record in view_records if record["output_name"] in selected_names]
    dropped = [{**record, "filtered_out": True} for record in view_records if record["output_name"] not in selected_names]
    return kept, dropped


def prepare_scene(
    *,
    config_path: str | Path | None = None,
    scene: str,
    num_intermediate_views: int | None = None,
    max_pairs: int | None = None,
    dust3r_variant: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    resolved = resolve_base_scene(config, scene, dust3r_variant)
    geometry_defaults = dict(config.defaults.get("geometry", {}))
    trajectory_manifest_path = build_trajectory_manifest(
        config=config,
        scene_name=resolved.scene_key,
        base_scene_dir=resolved.base_scene_dir,
        num_intermediate_views=num_intermediate_views,
        max_pairs=max_pairs,
        dust3r_variant=dust3r_variant,
        run_id=run_id,
    )
    trajectory_manifest_path = render_trajectory_coarse_guidance(
        config,
        trajectory_manifest_path,
        splat_radius=int(geometry_defaults.get("splat_radius", 2)),
    )
    trajectory = read_json(trajectory_manifest_path)
    return {
        "scene": resolved.scene_key,
        "scene_rel": str(resolved.scene_rel),
        "base_scene_dir": str(resolved.base_scene_dir),
        "source_type": resolved.source_type,
        "trajectory_manifest_path": str(trajectory_manifest_path),
        "coarse_guidance_root": trajectory.get("coarse_guidance_root"),
        "suggested_commands_if_missing": suggest_scene_build_commands(config, resolved.scene_key, dust3r_variant),
    }


def generate_pseudo_views(
    *,
    config_path: str | Path | None = None,
    trajectory_manifest_path: str | Path,
    prompt: str | None = None,
    dynami_crafter: dict[str, Any] | None = None,
    keep_ratio: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    adapter = _get_adapter(config)
    _DynamiCrafterInterpolationAdapter, extract_uniform_frames = _get_dynami_modules()

    trajectory = read_json(trajectory_manifest_path)
    run_dir = Path(trajectory["run_dir"])
    prompt_text = prompt or str(config.defaults.get("prompt"))
    dynami_defaults = dict(config.defaults.get("dynami_crafter", {}))
    dynami_defaults.update(dynami_crafter or {})
    dynami_defaults.setdefault("resolution", "384_512")
    geometry_defaults = dict(config.defaults.get("geometry", {}))
    filter_defaults = dict(config.defaults.get("pseudo_filter", {}))
    keep_ratio_value = float(keep_ratio if keep_ratio is not None else filter_defaults.get("keep_ratio", 0.8))
    min_keep_per_clip = int(filter_defaults.get("min_keep_per_clip", 1))
    renderer = CoarseGeometryRenderer.from_scene(
        project_root=config.project_root,
        base_scene_dir=trajectory["base_scene_dir"],
        splat_radius=int(geometry_defaults.get("splat_radius", 2)),
    )

    clips = []
    for pair in trajectory["pairs"]:
        clip_dir = run_dir / "pseudo_views" / pair["pair_id"]
        frame_dir = clip_dir / "frames"
        video_path = adapter.generate_clip(
            start_image_path=pair["start_image_path"],
            end_image_path=pair["end_image_path"],
            prompt=prompt_text,
            output_dir=clip_dir,
            steps=int(dynami_defaults.get("steps", 50)),
            cfg_scale=float(dynami_defaults.get("cfg_scale", 7.5)),
            eta=float(dynami_defaults.get("eta", 1.0)),
            fs=int(dynami_defaults.get("fs", 5)),
            seed=int(dynami_defaults.get("seed", 123)),
            resolution=str(dynami_defaults.get("resolution", "384_512")),
        )
        extracted = extract_uniform_frames(
            video_path=video_path,
            output_dir=frame_dir,
            num_frames=len(pair["intermediate_views"]),
            prefix=pair["pair_id"],
        )
        processed_views = []
        for view_record, frame_path in zip(pair["intermediate_views"], extracted):
            pseudo_rgb = load_rgb_image(frame_path)
            target_shape = (int(view_record["height"]), int(view_record["width"]))
            if pseudo_rgb.shape[:2] != target_shape:
                pseudo_rgb = resize_rgb_to_shape(pseudo_rgb, target_shape)
                save_rgb_image(frame_path, pseudo_rgb)
            if all(key in view_record for key in ("coarse_depth_path", "coarse_alpha_path", "coarse_normal_path", "coarse_rgb_path")):
                init_bundle = load_render_bundle(view_record)
            else:
                init_bundle = renderer.render_view(
                    project_root=config.project_root,
                    qvec=view_record["qvec"],
                    tvec=view_record["tvec"],
                    camera_model=str(view_record["camera_model"]),
                    camera_params=view_record["camera_params"],
                    width=int(view_record["width"]),
                    height=int(view_record["height"]),
                )
            init_reproj = compute_anchor_reprojection(
                project_root=config.project_root,
                pseudo_rgb=pseudo_rgb,
                target_qvec=view_record["qvec"],
                target_tvec=view_record["tvec"],
                target_camera_model=str(view_record["camera_model"]),
                target_camera_params=view_record["camera_params"],
                target_depth=init_bundle.depth,
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
            init_metrics = score_pseudo_frame(
                pseudo_rgb=pseudo_rgb,
                coarse_rgb=init_bundle.rgb,
                reprojection_error=init_reproj["mean_error"],
            )

            pose_meta = refine_pose_with_pnp(
                project_root=config.project_root,
                pseudo_rgb=pseudo_rgb,
                coarse_bundle=init_bundle,
                init_qvec=view_record["qvec"],
                init_tvec=view_record["tvec"],
                camera_model=str(view_record["camera_model"]),
                camera_params=view_record["camera_params"],
                min_matches=int(filter_defaults.get("pose_min_matches", 20)),
            )

            chosen_qvec = list(view_record["qvec"])
            chosen_tvec = list(view_record["tvec"])
            chosen_bundle = init_bundle
            chosen_reproj = init_reproj
            chosen_metrics = init_metrics
            pose_refined = False

            if pose_meta["success"] and int(pose_meta["num_inliers"]) >= int(filter_defaults.get("pose_min_inliers", 12)):
                refined_bundle = renderer.render_view(
                    project_root=config.project_root,
                    qvec=pose_meta["qvec"],
                    tvec=pose_meta["tvec"],
                    camera_model=str(view_record["camera_model"]),
                    camera_params=view_record["camera_params"],
                    width=int(view_record["width"]),
                    height=int(view_record["height"]),
                )
                refined_reproj = compute_anchor_reprojection(
                    project_root=config.project_root,
                    pseudo_rgb=pseudo_rgb,
                    target_qvec=pose_meta["qvec"],
                    target_tvec=pose_meta["tvec"],
                    target_camera_model=str(view_record["camera_model"]),
                    target_camera_params=view_record["camera_params"],
                    target_depth=refined_bundle.depth,
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
                refined_metrics = score_pseudo_frame(
                    pseudo_rgb=pseudo_rgb,
                    coarse_rgb=refined_bundle.rgb,
                    reprojection_error=refined_reproj["mean_error"],
                )
                if float(refined_metrics["frame_score"]) <= float(init_metrics["frame_score"]):
                    chosen_qvec = list(pose_meta["qvec"])
                    chosen_tvec = list(pose_meta["tvec"])
                    chosen_bundle = refined_bundle
                    chosen_reproj = refined_reproj
                    chosen_metrics = refined_metrics
                    pose_refined = True

            coarse_paths = save_render_bundle(chosen_bundle, clip_dir / "coarse" / Path(view_record["output_name"]).stem)
            processed_views.append(
                {
                    **view_record,
                    "frame_path": str(frame_path),
                    "initial_qvec": list(view_record["qvec"]),
                    "initial_tvec": list(view_record["tvec"]),
                    "qvec": chosen_qvec,
                    "tvec": chosen_tvec,
                    "pose_refined": pose_refined,
                    "pose_refine_success": bool(pose_meta["success"]),
                    "pose_matches": int(pose_meta["num_matches"]),
                    "pose_inliers": int(pose_meta["num_inliers"]),
                    "reprojection_valid_ratio": float(np.mean(chosen_reproj["valid_mask"] > 0)),
                    **coarse_paths,
                    **chosen_metrics,
                }
            )

        kept_views, filtered_views = _split_kept_views(processed_views, keep_ratio_value, min_keep_per_clip)
        clips.append(
            {
                "pair_id": pair["pair_id"],
                "video_path": str(video_path),
                "frames_dir": str(frame_dir),
                "assigned_views": kept_views,
                "filtered_views": filtered_views,
                "num_generated_frames": len(processed_views),
                "num_kept_frames": len(kept_views),
            }
        )

    pseudo_manifest_path = write_json(
        run_dir / "pseudo_views" / "pseudo_manifest.json",
        {
            "scene": trajectory["scene"],
            "run_id": trajectory["run_id"],
            "run_dir": trajectory["run_dir"],
            "trajectory_manifest_path": str(Path(trajectory_manifest_path).expanduser()),
            "prompt": prompt_text,
            "keep_ratio": keep_ratio_value,
            "dynami_crafter": {
                "resolution": str(dynami_defaults.get("resolution", "384_512")),
                "steps": int(dynami_defaults.get("steps", 50)),
                "cfg_scale": float(dynami_defaults.get("cfg_scale", 7.5)),
                "eta": float(dynami_defaults.get("eta", 1.0)),
                "fs": int(dynami_defaults.get("fs", 5)),
                "seed": int(dynami_defaults.get("seed", 123)),
            },
            "clips": clips,
        },
    )
    return {
        "scene": trajectory["scene"],
        "run_id": trajectory["run_id"],
        "pseudo_manifest_path": str(pseudo_manifest_path),
        "num_clips": len(clips),
    }


def build_confidence(
    *,
    config_path: str | Path | None = None,
    pseudo_manifest_path: str | Path,
    border_margin_ratio: float | None = None,
    flow_sigma: float | None = None,
    anchor_sigma: float | None = None,
    reproj_sigma: float | None = None,
    alpha_threshold: float | None = None,
    boundary_percentile: float | None = None,
    boundary_dilate: int | None = None,
    feature_backend: str | None = None,
    feature_sigma: float | None = None,
    enable_clip_consistency: bool | None = None,
    enable_patch_pruning: bool | None = None,
    patch_size: int | None = None,
    patch_threshold: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    defaults = dict(config.defaults.get("confidence", {}))
    feature_defaults = dict(config.defaults.get("feature", {}))
    consistency_defaults = dict(config.defaults.get("consistency", {}))
    ablation_defaults = dict(config.defaults.get("ablation", {}))
    manifest_path = build_confidence_manifest(
        config=config,
        pseudo_manifest_path=pseudo_manifest_path,
        border_margin_ratio=float(defaults.get("border_margin_ratio", 0.0) if border_margin_ratio is None else border_margin_ratio),
        flow_sigma=float(flow_sigma or defaults.get("flow_sigma", 1.5)),
        anchor_sigma=float(anchor_sigma or defaults.get("anchor_sigma", 0.25)),
        reproj_sigma=float(reproj_sigma or defaults.get("reproj_sigma", 0.2)),
        alpha_threshold=float(alpha_threshold or defaults.get("alpha_threshold", 0.05)),
        boundary_percentile=float(boundary_percentile or defaults.get("boundary_percentile", 90.0)),
        boundary_dilate=int(boundary_dilate or defaults.get("boundary_dilate", 3)),
        feature_backend=str(feature_backend or feature_defaults.get("backend", "dust3r")),
        feature_sigma=float(feature_sigma or feature_defaults.get("sigma", 0.35)),
        feature_stride=int(feature_defaults.get("stride", 2)),
        feature_cache=bool(feature_defaults.get("cache", True)),
        enable_clip_consistency=bool(
            consistency_defaults.get("enable_clip_consistency", True)
            if enable_clip_consistency is None
            else enable_clip_consistency
        ),
        enable_patch_pruning=bool(
            consistency_defaults.get("enable_patch_pruning", True)
            if enable_patch_pruning is None
            else enable_patch_pruning
        ),
        patch_size=int(patch_size or consistency_defaults.get("patch_size", 16)),
        patch_threshold=float(patch_threshold or consistency_defaults.get("patch_threshold", 0.25)),
        patch_low_weight=float(consistency_defaults.get("patch_low_weight", 0.15)),
        patch_min_keep_ratio=float(consistency_defaults.get("patch_min_keep_ratio", 0.1)),
        ablation={key: bool(value) for key, value in ablation_defaults.items()},
    )
    return {"confidence_manifest_path": str(manifest_path)}


def build_hybrid(
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
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    trajectory = read_json(trajectory_manifest_path)
    resolved_hybrid_name = hybrid_name or f"{trajectory['scene']}_{trajectory['run_id']}_hybrid"
    hybrid_manifest_path = build_hybrid_scene(
        config=config,
        trajectory_manifest_path=trajectory_manifest_path,
        pseudo_manifest_path=pseudo_manifest_path,
        confidence_manifest_path=confidence_manifest_path,
        hybrid_name=resolved_hybrid_name,
    )
    response = {
        "hybrid_manifest_path": str(hybrid_manifest_path),
        "hybrid_scene_dir": str(Path(read_json(hybrid_manifest_path)["hybrid_scene_dir"]).expanduser()),
    }
    if train:
        response["training"] = launch_training(
            config=config,
            scene_dir=response["hybrid_scene_dir"],
            output_tag=output_tag or resolved_hybrid_name,
            iterations=iterations,
            confidence_manifest_path=confidence_manifest_path,
        )
    return response


def evaluate_part3(
    *,
    config_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_tag: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    target = model_dir
    if not target:
        if not output_tag:
            raise ValueError("Either model_dir or output_tag is required for evaluation.")
        target = config.workspace_root / "3dgs_outputs" / output_tag
    return {"evaluation": launch_evaluation(config, target)}


def run_part3_pipeline(
    *,
    config_path: str | Path | None = None,
    scene: str,
    prompt: str | None = None,
    num_intermediate_views: int | None = None,
    max_pairs: int | None = None,
    dust3r_variant: str | None = None,
    run_id: str | None = None,
    dynami_crafter: dict[str, Any] | None = None,
    border_margin_ratio: float | None = None,
    flow_sigma: float | None = None,
    anchor_sigma: float | None = None,
    hybrid_name: str | None = None,
    train: bool = False,
    evaluate: bool = False,
    output_tag: str | None = None,
    iterations: int | None = None,
    keep_ratio: float | None = None,
    reproj_sigma: float | None = None,
    alpha_threshold: float | None = None,
    boundary_percentile: float | None = None,
    boundary_dilate: int | None = None,
    feature_backend: str | None = None,
    feature_sigma: float | None = None,
    enable_clip_consistency: bool | None = None,
    enable_patch_pruning: bool | None = None,
    patch_size: int | None = None,
    patch_threshold: float | None = None,
) -> dict[str, Any]:
    scene_result = prepare_scene(
        config_path=config_path,
        scene=scene,
        num_intermediate_views=num_intermediate_views,
        max_pairs=max_pairs,
        dust3r_variant=dust3r_variant,
        run_id=run_id,
    )
    pseudo_result = generate_pseudo_views(
        config_path=config_path,
        trajectory_manifest_path=scene_result["trajectory_manifest_path"],
        prompt=prompt,
        dynami_crafter=dynami_crafter,
        keep_ratio=keep_ratio,
    )
    confidence_result = build_confidence(
        config_path=config_path,
        pseudo_manifest_path=pseudo_result["pseudo_manifest_path"],
        border_margin_ratio=border_margin_ratio,
        flow_sigma=flow_sigma,
        anchor_sigma=anchor_sigma,
        reproj_sigma=reproj_sigma,
        alpha_threshold=alpha_threshold,
        boundary_percentile=boundary_percentile,
        boundary_dilate=boundary_dilate,
        feature_backend=feature_backend,
        feature_sigma=feature_sigma,
        enable_clip_consistency=enable_clip_consistency,
        enable_patch_pruning=enable_patch_pruning,
        patch_size=patch_size,
        patch_threshold=patch_threshold,
    )
    hybrid_result = build_hybrid(
        config_path=config_path,
        trajectory_manifest_path=scene_result["trajectory_manifest_path"],
        pseudo_manifest_path=pseudo_result["pseudo_manifest_path"],
        confidence_manifest_path=confidence_result["confidence_manifest_path"],
        hybrid_name=hybrid_name,
        train=train,
        output_tag=output_tag,
        iterations=iterations,
    )

    evaluation_result = None
    if evaluate:
        model_dir = None
        if "training" in hybrid_result:
            model_dir = hybrid_result["training"]["output_dir"]
        evaluation_result = evaluate_part3(
            config_path=config_path,
            model_dir=model_dir,
            output_tag=output_tag,
        )

    return {
        "scene": scene_result,
        "pseudo_view": pseudo_result,
        "confidence": confidence_result,
        "hybrid": hybrid_result,
        "evaluation": evaluation_result,
    }
