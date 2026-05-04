#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "part3" / "src"))

from eval_ate_rmse import align_points, load_cameras_json_candidates, load_gt_candidates, umeyama_alignment
from part3_stack.geometry import _load_colmap_io

try:
    import torch
    import torchvision.transforms.functional as tf
    from lpipsPyTorch import lpips
    from utils.loss_utils import ssim

    TORCH_METRICS_AVAILABLE = True
except Exception:
    TORCH_METRICS_AVAILABLE = False


def load_results(model_dir: Path) -> dict[str, Any]:
    results_path = model_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.json not found under {model_dir}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def load_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    json_path = Path(path).expanduser()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def first_method_metrics(results: dict[str, Any], requested_method: str = "") -> tuple[str, dict[str, float]]:
    if not results:
        raise ValueError("Empty results.json")

    def is_metrics_dict(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        metric_keys = {"PSNR", "SSIM", "LPIPS", "psnr", "ssim", "lpips"}
        return any(key in value for key in metric_keys) and all(
            isinstance(item, (int, float)) for item in value.values()
        )

    def as_float_metrics(value: dict[str, Any]) -> dict[str, float]:
        return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))}

    if is_metrics_dict(results):
        if requested_method and requested_method != "results":
            raise ValueError(f"Requested method '{requested_method}' not found in flat results.json")
        return "results", as_float_metrics(results)

    if requested_method:
        payload = results.get(requested_method)
        if is_metrics_dict(payload):
            return str(requested_method), as_float_metrics(payload)
        raise ValueError(f"Requested method '{requested_method}' not found in results.json")

    for name, payload in results.items():
        if is_metrics_dict(payload):
            return str(name), as_float_metrics(payload)
        if not isinstance(payload, dict):
            continue
        for method_name, metrics in payload.items():
            if is_metrics_dict(metrics):
                return f"{name}/{method_name}", as_float_metrics(metrics)
    raise ValueError("No PSNR/SSIM/LPIPS metrics found in results.json")


def rotation_angle(rot: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)))


def load_colmap_c2w_poses(scene_dir: Path) -> dict[str, np.ndarray]:
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = scene_dir / "sparse"
    colmap_io = _load_colmap_io(PROJECT_ROOT)
    _cameras, images, _points3d = colmap_io.read_model(str(sparse_dir), ext="")
    poses: dict[str, np.ndarray] = {}
    for image in images.values():
        if str(image.name).startswith("pseudo_pair_"):
            continue
        rot_w2c = image.qvec2rotmat()
        center = -rot_w2c.T @ np.asarray(image.tvec, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot_w2c.T
        pose[:3, 3] = center
        poses[Path(image.name).name] = pose
        poses[Path(image.name).stem] = pose
    return poses


def load_gt_candidates_from_arg(eval_meta: str, fallback_scene_dir: Path | None) -> dict[str, dict[str, np.ndarray]]:
    if eval_meta:
        path = Path(eval_meta).expanduser()
        if path.is_file():
            return load_cameras_json_candidates(path)
        if path.is_dir() and (path / "cameras.json").is_file():
            return load_cameras_json_candidates(path / "cameras.json")
        if path.is_dir() and (path / "eval_meta" / "cameras.json").is_file():
            return load_cameras_json_candidates(path / "eval_meta" / "cameras.json")
        raise FileNotFoundError(f"Could not resolve eval metadata from: {path}")
    if fallback_scene_dir is None:
        return {}
    return load_gt_candidates(fallback_scene_dir, "auto")


def evaluate_scene_pose(scene_dir: str, eval_meta: str = "") -> dict[str, Any] | None:
    if not scene_dir:
        return None
    scene_path = Path(scene_dir).expanduser()
    if not scene_path.is_dir():
        return None
    est_poses = load_colmap_c2w_poses(scene_path)
    gt_candidates = load_gt_candidates_from_arg(eval_meta, scene_path if not eval_meta else None)

    best: dict[str, Any] | None = None
    for convention, gt_poses in gt_candidates.items():
        common_names = sorted(name for name in est_poses if name in gt_poses and "." in name)
        if len(common_names) < 2:
            continue
        est_xyz = np.asarray([est_poses[name][:3, 3] for name in common_names], dtype=np.float64)
        gt_xyz = np.asarray([gt_poses[name][:3, 3] for name in common_names], dtype=np.float64)
        scale, rot_align, trans_align = umeyama_alignment(est_xyz, gt_xyz)
        aligned_xyz = align_points(est_xyz, scale, rot_align, trans_align)
        residuals = gt_xyz - aligned_xyz
        ate_per_frame = np.linalg.norm(residuals, axis=1)

        aligned_poses: dict[str, np.ndarray] = {}
        for name, center in zip(common_names, aligned_xyz):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rot_align @ est_poses[name][:3, :3]
            pose[:3, 3] = center
            aligned_poses[name] = pose

        rpe_trans = []
        rpe_rot = []
        for left, right in zip(common_names[:-1], common_names[1:]):
            gt_rel = np.linalg.inv(gt_poses[left]) @ gt_poses[right]
            est_rel = np.linalg.inv(aligned_poses[left]) @ aligned_poses[right]
            err = np.linalg.inv(gt_rel) @ est_rel
            rpe_trans.append(float(np.linalg.norm(err[:3, 3])))
            rpe_rot.append(rotation_angle(err[:3, :3]))

        candidate = {
            "scene_dir": str(scene_path),
            "convention": convention,
            "num_matches": len(common_names),
            "ate_rmse": float(np.sqrt(np.mean(ate_per_frame * ate_per_frame))),
            "ate_mean": float(np.mean(ate_per_frame)),
            "ate_median": float(np.median(ate_per_frame)),
            "rpe_translation_rmse": float(np.sqrt(np.mean(np.square(rpe_trans)))) if rpe_trans else 0.0,
            "rpe_rotation_rmse_rad": float(np.sqrt(np.mean(np.square(rpe_rot)))) if rpe_rot else 0.0,
            "scale": float(scale),
        }
        if best is None or candidate["ate_rmse"] < best["ate_rmse"]:
            best = candidate
    return best


def summarize_confidence_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    clips = manifest.get("clips", [])
    records = [record for clip in clips for record in clip.get("mask_records", [])]
    clip_metrics = [clip.get("clip_metrics", {}) for clip in clips if clip.get("clip_metrics")]

    def mean_key(items: list[dict[str, Any]], key: str) -> float:
        values = [float(item[key]) for item in items if key in item]
        return float(np.mean(values)) if values else 0.0

    return {
        "num_clips": len(clips),
        "num_records": len(records),
        "mean_confidence": mean_key(records, "mean_confidence"),
        "mean_raw_confidence": mean_key(records, "mean_raw_confidence"),
        "mean_reprojection_error": mean_key(records, "mean_reprojection_error"),
        "mean_reprojection_valid_ratio": mean_key(records, "reprojection_valid_ratio"),
        "mean_reprojection_confidence": mean_key(records, "mean_reprojection_confidence"),
        "mean_feature_confidence": mean_key(records, "mean_feature_confidence"),
        "mean_temporal_confidence": mean_key(records, "mean_temporal_confidence"),
        "mean_soft_confidence": mean_key(records, "mean_soft_confidence"),
        "mean_hard_validity": mean_key(records, "mean_hard_validity"),
        "mean_patch_keep_ratio": mean_key(records, "patch_keep_ratio"),
        "mean_clip_score": mean_key(clip_metrics, "clip_score"),
        "mean_pose_translation_std": mean_key(clip_metrics, "pose_translation_std"),
        "mean_pose_rotation_std": mean_key(clip_metrics, "pose_rotation_std"),
        "ablation": manifest.get("ablation", {}),
        "feature": manifest.get("feature", {}),
        "consistency": manifest.get("consistency", {}),
    }


def load_per_view(model_dir: Path, requested_method: str = "") -> tuple[str, dict[str, dict[str, float]]] | tuple[str, None]:
    path = model_dir / "per_view.json"
    if not path.is_file():
        return "", None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return "", None
    if requested_method:
        method_payload = payload.get(requested_method)
        if method_payload is None:
            raise ValueError(f"Requested per-view method '{requested_method}' not found in {path}")
        return requested_method, method_payload
    method = next(iter(payload))
    return method, payload[method]


def load_view_mapping(model_dir: Path, method: str, split: str = "test") -> dict[str, dict[str, Any]]:
    path = model_dir / split / method / "view_mapping.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def split_full_metrics(per_view: dict[str, dict[str, float]] | None, mapping: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not per_view or not mapping:
        return {}
    groups = {
        "real_only_full_metrics": [],
        "pseudo_only_full_metrics": [],
        "all_full_metrics": [],
    }
    for output_name, meta in mapping.items():
        target = "pseudo_only_full_metrics" if bool(meta.get("is_pseudo", False)) else "real_only_full_metrics"
        groups[target].append(output_name)
        groups["all_full_metrics"].append(output_name)

    summary: dict[str, Any] = {}
    for group_name, names in groups.items():
        summary[group_name] = {"count": len(names), "view_names": [mapping[name].get("image_name", name) for name in names]}
        for metric in ("PSNR", "SSIM", "LPIPS"):
            values = [float(per_view.get(metric, {}).get(name)) for name in names if name in per_view.get(metric, {})]
            summary[group_name][metric] = _mean(values)
    summary["pseudo_full_metrics"] = dict(summary["pseudo_only_full_metrics"])
    return summary


def _alpha_tensor(scene_dir: Path, image_name: str, height: int, width: int, device) -> Any:
    image_path = scene_dir / "images" / image_name
    if not image_path.is_file() and not image_name.endswith(".png"):
        image_path = scene_dir / "images" / f"{image_name}.png"
    if image_path.is_file():
        image = Image.open(image_path)
        if image.mode == "RGBA":
            mask = tf.to_tensor(image.getchannel("A"))[None].to(device)
        else:
            mask = torch.ones((1, 1, height, width), device=device)
    else:
        mask = torch.ones((1, 1, height, width), device=device)
    if tuple(mask.shape[-2:]) != (height, width):
        mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _psnr_from_mse(mse_value) -> float:
    return float((20.0 * torch.log10(torch.tensor(1.0, device=mse_value.device) / torch.sqrt(mse_value.clamp_min(1e-8)))).item())


def rendered_full_metrics(
    model_dir: Path,
    method: str,
    mapping: dict[str, dict[str, Any]],
    *,
    split: str,
    pseudo_only: bool,
) -> dict[str, Any]:
    if not TORCH_METRICS_AVAILABLE or not mapping:
        return {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    render_dir = model_dir / split / method / "renders"
    gt_dir = model_dir / split / method / "gt"
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    l1s: list[float] = []
    names: list[str] = []

    for output_name, meta in mapping.items():
        is_pseudo = bool(meta.get("is_pseudo", False))
        if pseudo_only and not is_pseudo:
            continue
        render_path = render_dir / output_name
        gt_path = gt_dir / output_name
        if not render_path.is_file() or not gt_path.is_file():
            continue
        pred = tf.to_tensor(Image.open(render_path)).unsqueeze(0)[:, :3].to(device)
        target = tf.to_tensor(Image.open(gt_path)).unsqueeze(0)[:, :3].to(device)
        diff = pred - target
        mse_value = (diff * diff).mean()
        psnrs.append(_psnr_from_mse(mse_value))
        l1s.append(float(diff.abs().mean().item()))
        ssims.append(float(ssim(pred, target).mean().item()))
        try:
            lpipss.append(float(lpips(pred, target, net_type="vgg").mean().item()))
        except Exception:
            pass
        names.append(str(meta.get("image_name", output_name)))

    return {
        "count": len(psnrs),
        "view_names": names,
        "PSNR": _mean(psnrs),
        "SSIM": _mean(ssims),
        "LPIPS": _mean(lpipss),
        "L1": _mean(l1s),
        "split": split,
    }


def masked_metrics(
    model_dir: Path,
    scene_dir: Path,
    method: str,
    mapping: dict[str, dict[str, Any]],
    *,
    split: str,
    pseudo_only: bool,
    normalize: bool,
) -> dict[str, Any]:
    if not TORCH_METRICS_AVAILABLE or not mapping or not scene_dir.is_dir():
        return {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    render_dir = model_dir / split / method / "renders"
    gt_dir = model_dir / split / method / "gt"
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    l1s: list[float] = []
    mask_means: list[float] = []
    names: list[str] = []

    for output_name, meta in mapping.items():
        is_pseudo = bool(meta.get("is_pseudo", False))
        if pseudo_only and not is_pseudo:
            continue
        render_path = render_dir / output_name
        gt_path = gt_dir / output_name
        if not render_path.is_file() or not gt_path.is_file():
            continue
        pred = tf.to_tensor(Image.open(render_path)).unsqueeze(0)[:, :3].to(device)
        target = tf.to_tensor(Image.open(gt_path)).unsqueeze(0)[:, :3].to(device)
        mask = _alpha_tensor(scene_dir, str(meta.get("image_name", output_name)), pred.shape[-2], pred.shape[-1], device)
        weight = mask.expand_as(pred)
        diff = pred - target
        masked_pred = pred * mask
        masked_target = target * mask
        if normalize:
            denom = weight.sum().clamp_min(1e-8)
            mse_value = ((diff * diff) * weight).sum() / denom
            l1_value = (diff.abs() * weight).sum() / denom
            lpips_norm = mask.mean().clamp_min(1e-6)
        else:
            mse_value = ((diff * mask) ** 2).mean()
            l1_value = (diff.abs() * mask).mean()
            lpips_norm = torch.ones((), device=device)
        psnrs.append(_psnr_from_mse(mse_value))
        l1s.append(float(l1_value.item()))
        ssims.append(float(ssim(masked_pred, masked_target).mean().item()))
        try:
            lpipss.append(float((lpips(masked_pred, masked_target, net_type="vgg") / lpips_norm).mean().item()))
        except Exception:
            pass
        mask_means.append(float(mask.mean().item()))
        names.append(str(meta.get("image_name", output_name)))

    return {
        "count": len(psnrs),
        "view_names": names,
        "PSNR": _mean(psnrs),
        "SSIM": _mean(ssims),
        "LPIPS": _mean(lpipss),
        "L1": _mean(l1s),
        "mean_mask": _mean(mask_means),
        "normalized": bool(normalize),
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Sparse Only vs Sparse+Generated metrics for Part 3")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline 3DGS model dir with results.json")
    parser.add_argument("--part3", type=str, required=True, help="Part 3 model dir with results.json")
    parser.add_argument("--baseline-scene-dir", type=str, default="", help="Baseline sparse-only COLMAP/3DGS scene dir for ATE/RPE")
    parser.add_argument("--part3-scene-dir", type=str, default="", help="Part 3 hybrid COLMAP/3DGS scene dir for ATE/RPE")
    parser.add_argument("--baseline-eval-meta", type=str, default="", help="GT eval_meta dir or cameras.json for pose metrics")
    parser.add_argument("--part3-hybrid-manifest", type=str, default="", help="Optional part3_hybrid_manifest.json for pseudo-view quality summary")
    parser.add_argument("--part3-confidence-manifest", type=str, default="", help="Optional confidence_manifest.json for pseudo consistency summary")
    parser.add_argument("--method", type=str, default="", help="Explicit render method/checkpoint name such as ours_15000 or ours_30000")
    parser.add_argument("--baseline-method", type=str, default="", help="Optional override for baseline method name")
    parser.add_argument("--part3-method", type=str, default="", help="Optional override for part3 method name")
    parser.add_argument("--output", type=str, default="", help="Optional output summary json path")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline).expanduser()
    part3_dir = Path(args.part3).expanduser()

    baseline_method_name = args.baseline_method or args.method
    part3_method_name = args.part3_method or args.method

    baseline_method, baseline_metrics = first_method_metrics(load_results(baseline_dir), baseline_method_name)
    part3_method, part3_metrics = first_method_metrics(load_results(part3_dir), part3_method_name)

    summary = {
        "baseline_dir": str(baseline_dir),
        "part3_dir": str(part3_dir),
        "baseline_method": baseline_method,
        "part3_method": part3_method,
        "baseline": baseline_metrics,
        "part3": part3_metrics,
        "delta": {
            key: float(part3_metrics[key] - baseline_metrics[key])
            for key in baseline_metrics
            if key in part3_metrics
        },
    }
    hybrid_manifest = load_optional_json(args.part3_hybrid_manifest)
    if hybrid_manifest is not None:
        summary["part3_hybrid_summary"] = hybrid_manifest.get("summary", {})
        if not args.part3_scene_dir:
            args.part3_scene_dir = hybrid_manifest.get("hybrid_scene_dir", "")

    confidence_manifest = load_optional_json(args.part3_confidence_manifest)
    confidence_summary = summarize_confidence_manifest(confidence_manifest)
    if confidence_summary:
        summary["part3_pseudo_consistency"] = confidence_summary

    per_view_method, per_view = load_per_view(part3_dir, part3_method)
    mapping_method = part3_method or per_view_method
    test_view_mapping = load_view_mapping(part3_dir, mapping_method, split="test")
    train_view_mapping = load_view_mapping(part3_dir, mapping_method, split="train")
    split_metrics = split_full_metrics(per_view, test_view_mapping)
    if train_view_mapping:
        pseudo_train_full = rendered_full_metrics(
            part3_dir,
            mapping_method,
            train_view_mapping,
            split="train",
            pseudo_only=True,
        )
        if pseudo_train_full.get("count", 0):
            split_metrics["pseudo_only_full_metrics"] = pseudo_train_full
            split_metrics["pseudo_full_metrics"] = dict(pseudo_train_full)
    if split_metrics:
        summary["part3_split_metrics"] = split_metrics
    part3_main_metrics = part3_metrics
    part3_main_source = "results_json_overall"
    real_full_metrics = split_metrics.get("real_only_full_metrics", {}) if split_metrics else {}
    if real_full_metrics.get("count", 0):
        candidate = {
            key: float(value)
            for key, value in real_full_metrics.items()
            if key in {"PSNR", "SSIM", "LPIPS"} and value is not None
        }
        if candidate:
            part3_main_metrics = candidate
            part3_main_source = "real_only_full_metrics"
    summary["part3_main_metrics"] = part3_main_metrics
    summary["part3_main_metric_source"] = part3_main_source
    summary["main_delta"] = {
        key: float(part3_main_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
        if key in part3_main_metrics
    }
    part3_scene_for_masks = Path(args.part3_scene_dir).expanduser() if args.part3_scene_dir else None
    pseudo_mapping = train_view_mapping or test_view_mapping
    pseudo_split = "train" if train_view_mapping else "test"
    if part3_scene_for_masks is not None and pseudo_mapping:
        normalized = masked_metrics(
            part3_dir,
            part3_scene_for_masks,
            mapping_method,
            pseudo_mapping,
            split=pseudo_split,
            pseudo_only=True,
            normalize=True,
        )
        unnormalized = masked_metrics(
            part3_dir,
            part3_scene_for_masks,
            mapping_method,
            pseudo_mapping,
            split=pseudo_split,
            pseudo_only=True,
            normalize=False,
        )
        if normalized:
            summary["pseudo_masked_normalized_metrics"] = normalized
        if unnormalized:
            summary["pseudo_masked_unnormalized_debug_metrics"] = unnormalized
            summary["pseudo_masked_unnormalized_debug"] = unnormalized

    baseline_pose = evaluate_scene_pose(args.baseline_scene_dir, args.baseline_eval_meta)
    part3_pose = evaluate_scene_pose(args.part3_scene_dir, args.baseline_eval_meta)
    if baseline_pose is not None:
        summary["baseline_pose"] = baseline_pose
    if part3_pose is not None:
        summary["part3_pose"] = part3_pose
    if baseline_pose is not None and part3_pose is not None:
        summary["pose_delta"] = {
            key: float(part3_pose[key] - baseline_pose[key])
            for key in ("ate_rmse", "rpe_translation_rmse", "rpe_rotation_rmse_rad")
            if key in baseline_pose and key in part3_pose
        }

    print(f"Primary metrics source: {part3_main_source}")
    print("Metric    Baseline       Part3Main   Delta")
    for key in ("PSNR", "SSIM", "LPIPS"):
        if key not in summary["main_delta"]:
            continue
        print(
            f"{key:<8} "
            f"{baseline_metrics[key]:>10.4f} "
            f"{part3_main_metrics[key]:>10.4f} "
            f"{summary['main_delta'][key]:>10.4f}"
        )
    if part3_main_source != "results_json_overall":
        print("\nLegacy Part3 results.json overall")
        for key in ("PSNR", "SSIM", "LPIPS"):
            if key in part3_metrics:
                print(f"{key}: {part3_metrics[key]}")

    if "part3_hybrid_summary" in summary:
        hybrid_summary = summary["part3_hybrid_summary"]
        print("\nPart3 Hybrid Summary")
        for key in (
            "num_generated_views",
            "num_kept_views",
            "num_filtered_views",
            "mean_reprojection_error",
            "mean_confidence",
            "mean_feature_confidence",
            "mean_patch_keep_ratio",
            "mean_clip_score",
        ):
            if key in hybrid_summary:
                print(f"{key}: {hybrid_summary[key]}")

    if "part3_pseudo_consistency" in summary:
        print("\nPart3 Pseudo Consistency")
        pseudo = summary["part3_pseudo_consistency"]
        for key in (
            "mean_clip_score",
            "mean_reprojection_error",
            "mean_reprojection_valid_ratio",
            "mean_feature_confidence",
            "mean_soft_confidence",
            "mean_hard_validity",
            "mean_patch_keep_ratio",
            "mean_pose_translation_std",
            "mean_pose_rotation_std",
        ):
            if key in pseudo:
                print(f"{key}: {pseudo[key]}")

    if "part3_split_metrics" in summary:
        print("\nPart3 Split Full-Image Metrics")
        for label, metrics in summary["part3_split_metrics"].items():
            print(
                f"{label}: count={metrics['count']}, "
                f"PSNR={metrics.get('PSNR')}, SSIM={metrics.get('SSIM')}, LPIPS={metrics.get('LPIPS')}"
            )

    for label in ("pseudo_masked_normalized_metrics", "pseudo_masked_unnormalized_debug_metrics"):
        if label in summary:
            metrics = summary[label]
            print(
                f"\n{label}: count={metrics['count']}, "
                f"PSNR={metrics.get('PSNR')}, SSIM={metrics.get('SSIM')}, "
                f"LPIPS={metrics.get('LPIPS')}, mean_mask={metrics.get('mean_mask')}"
            )

    if "baseline_pose" in summary or "part3_pose" in summary:
        print("\nPose Metrics")
        for label, metrics in (("baseline", summary.get("baseline_pose")), ("part3", summary.get("part3_pose"))):
            if not metrics:
                continue
            print(
                f"{label}: ATE={metrics['ate_rmse']:.6f}, "
                f"RPE_t={metrics['rpe_translation_rmse']:.6f}, "
                f"RPE_r={metrics['rpe_rotation_rmse_rad']:.6f} rad"
            )

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
