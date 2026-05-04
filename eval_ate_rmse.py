#!/usr/bin/env python3
"""
Evaluate DUSt3R pose estimates with ATE RMSE after trajectory alignment.

This script compares `results_dust3r/<scene>/cam2world.npy` against the GT poses
saved in `data_p2_sparse/<scene>/eval_meta/`.

Supported GT formats:
  - DL3DV / Re10k: eval_meta/cameras.json
  - Waymo-405841:  eval_meta/gt/*.txt or eval_meta/gt_old/*.txt

The script aligns camera centers with a Sim(3) transform (Umeyama) and reports
ATE RMSE per scene.

Examples:
  python eval_ate_rmse.py \
    --results-root results_dust3r \
    --data-root data_p2_sparse

  python eval_ate_rmse.py \
    --input-dir results_dust3r/DL3DV-2 \
    --scene-dir data_p2_sparse/DL3DV-2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Sim(3)-aligned ATE RMSE for DUSt3R poses.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Single DUSt3R scene output directory, e.g. results_dust3r/DL3DV-2",
    )
    group.add_argument(
        "--results-root",
        type=str,
        default=str(SCRIPT_DIR / "results_dust3r"),
        help="Root containing DUSt3R scene outputs.",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        default="",
        help="Matching sparse data scene directory for --input-dir mode.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(SCRIPT_DIR / "data_p2_sparse"),
        help="Root containing sparse Part 2 scenes and eval_meta.",
    )
    parser.add_argument(
        "--only-scene",
        type=str,
        default="",
        help="Only evaluate scenes whose relative path contains this substring.",
    )
    parser.add_argument(
        "--waymo-source",
        type=str,
        default="auto",
        choices=["auto", "gt", "gt_old"],
        help="Which Waymo GT matrix folder to use. 'auto' picks the best-matching candidate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "results_dust3r" / "ate_rmse_summary.json"),
        help="Path to the summary JSON file.",
    )
    return parser.parse_args()


def discover_scene_dirs(results_root: Path) -> list[Path]:
    return sorted(meta.parent for meta in results_root.rglob("dust3r_meta.json"))


def load_scene_meta(result_dir: Path) -> dict:
    with (result_dir / "dust3r_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_scene_rel(result_dir: Path, meta: dict, results_root: Path | None) -> Path:
    scene_rel = meta.get("scene_rel")
    if scene_rel:
        return Path(scene_rel)
    if results_root is not None:
        return result_dir.relative_to(results_root)
    return Path(result_dir.name)


def normalize_name(name: str) -> str:
    path = Path(name)
    return path.name


def add_pose_entry(target: dict[str, np.ndarray], name: str, c2w: np.ndarray) -> None:
    target[normalize_name(name)] = c2w
    target[Path(name).stem] = c2w


def quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return quat_xyzw_to_rotmat(np.array([x, y, z, w], dtype=np.float64))


def pose_from_rot_trans(rot: np.ndarray, trans: np.ndarray, mode: str) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    if mode == "c2w":
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        return pose
    if mode == "w2c":
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        return np.linalg.inv(pose)
    raise ValueError(f"Unsupported pose mode: {mode}")


def load_cameras_json_candidates(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    candidates = {
        "quat_xyzw+c2w": {},
        "quat_xyzw+w2c": {},
        "quat_wxyz+c2w": {},
        "quat_wxyz+w2c": {},
    }

    for record in records:
        name = record["image_name"]
        quat = np.asarray(record["cam_quat"], dtype=np.float64)
        trans = np.asarray(record["cam_trans"], dtype=np.float64)
        rot_xyzw = quat_xyzw_to_rotmat(quat)
        rot_wxyz = quat_wxyz_to_rotmat(quat)

        add_pose_entry(candidates["quat_xyzw+c2w"], name, pose_from_rot_trans(rot_xyzw, trans, "c2w"))
        add_pose_entry(candidates["quat_xyzw+w2c"], name, pose_from_rot_trans(rot_xyzw, trans, "w2c"))
        add_pose_entry(candidates["quat_wxyz+c2w"], name, pose_from_rot_trans(rot_wxyz, trans, "c2w"))
        add_pose_entry(candidates["quat_wxyz+w2c"], name, pose_from_rot_trans(rot_wxyz, trans, "w2c"))
    return candidates


def load_matrix_folder(path: Path) -> dict[str, np.ndarray]:
    entries: dict[str, np.ndarray] = {}
    for matrix_path in sorted(path.glob("*.txt")):
        matrix = np.loadtxt(matrix_path, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose matrix in {matrix_path}, got {matrix.shape}")
        add_pose_entry(entries, f"{matrix_path.stem}.png", matrix)
    return entries


def load_waymo_candidates(eval_meta_dir: Path, waymo_source: str) -> dict[str, dict[str, np.ndarray]]:
    folders: list[tuple[str, Path]] = []
    if waymo_source in {"auto", "gt"} and (eval_meta_dir / "gt").is_dir():
        folders.append(("gt", eval_meta_dir / "gt"))
    if waymo_source in {"auto", "gt_old"} and (eval_meta_dir / "gt_old").is_dir():
        folders.append(("gt_old", eval_meta_dir / "gt_old"))
    if not folders:
        raise FileNotFoundError(f"No Waymo GT pose folders found under: {eval_meta_dir}")

    candidates: dict[str, dict[str, np.ndarray]] = {}
    for label, folder in folders:
        direct = load_matrix_folder(folder)
        inverse = {name: np.linalg.inv(pose) for name, pose in direct.items()}
        candidates[f"{label}+c2w"] = direct
        candidates[f"{label}+w2c"] = inverse
    return candidates


def load_gt_candidates(scene_dir: Path, waymo_source: str) -> dict[str, dict[str, np.ndarray]]:
    eval_meta_dir = scene_dir / "eval_meta"
    cameras_json = eval_meta_dir / "cameras.json"
    if cameras_json.is_file():
        return load_cameras_json_candidates(cameras_json)
    return load_waymo_candidates(eval_meta_dir, waymo_source)


def load_estimated_poses(result_dir: Path, meta: dict) -> dict[str, np.ndarray]:
    cam2world = np.load(result_dir / "cam2world.npy")
    image_names = [Path(p).name for p in meta["image_paths"]]
    if cam2world.shape[0] != len(image_names):
        raise ValueError(
            f"Shape mismatch in {result_dir}: cam2world={cam2world.shape}, image_paths={len(image_names)}"
        )
    poses: dict[str, np.ndarray] = {}
    for image_name, pose in zip(image_names, cam2world):
        add_pose_entry(poses, image_name, pose)
    return poses


def umeyama_alignment(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source_xyz.shape != target_xyz.shape:
        raise ValueError(f"source and target shapes do not match: {source_xyz.shape} vs {target_xyz.shape}")
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {source_xyz.shape}")
    n = source_xyz.shape[0]
    if n < 2:
        raise ValueError("At least two matched poses are required for Sim(3) alignment.")

    src_mean = source_xyz.mean(axis=0)
    tgt_mean = target_xyz.mean(axis=0)
    src_centered = source_xyz - src_mean
    tgt_centered = target_xyz - tgt_mean

    cov = (tgt_centered.T @ src_centered) / n
    u, singular_values, vt = np.linalg.svd(cov)
    reflection = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection[-1, -1] = -1.0

    rot = u @ reflection @ vt
    var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
    if var_src <= 0:
        raise ValueError("Degenerate source trajectory for alignment.")
    scale = float(np.trace(np.diag(singular_values) @ reflection) / var_src)
    trans = tgt_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def align_points(source_xyz: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ source_xyz.T)).T + trans


def compute_rmse(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> tuple[float, np.ndarray]:
    residuals = gt_xyz - pred_xyz
    per_frame = np.linalg.norm(residuals, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return rmse, per_frame


def evaluate_scene(result_dir: Path, scene_dir: Path, waymo_source: str) -> dict:
    meta = load_scene_meta(result_dir)
    est_poses = load_estimated_poses(result_dir, meta)
    gt_candidates = load_gt_candidates(scene_dir, waymo_source)

    best: dict | None = None
    for convention, gt_poses in gt_candidates.items():
        common_names = sorted(name for name in est_poses if name in gt_poses and "." in name)
        if len(common_names) < 2:
            continue

        est_xyz = np.asarray([est_poses[name][:3, 3] for name in common_names], dtype=np.float64)
        gt_xyz = np.asarray([gt_poses[name][:3, 3] for name in common_names], dtype=np.float64)
        scale, rot, trans = umeyama_alignment(est_xyz, gt_xyz)
        aligned_est_xyz = align_points(est_xyz, scale, rot, trans)
        rmse, per_frame = compute_rmse(gt_xyz, aligned_est_xyz)

        candidate = {
            "scene_rel": meta.get("scene_rel", result_dir.name),
            "result_dir": str(result_dir.resolve()),
            "scene_dir": str(scene_dir.resolve()),
            "convention": convention,
            "num_matches": len(common_names),
            "ate_rmse": rmse,
            "mean_translation_error": float(np.mean(per_frame)),
            "median_translation_error": float(np.median(per_frame)),
            "scale": float(scale),
            "rotation_matrix": rot.tolist(),
            "translation": trans.tolist(),
            "matched_image_names": common_names,
        }
        if best is None or candidate["ate_rmse"] < best["ate_rmse"]:
            best = candidate

    if best is None:
        raise RuntimeError(f"Could not find at least two matched poses between {result_dir} and {scene_dir}")
    return best


def iter_jobs(args: argparse.Namespace) -> Iterable[tuple[Path, Path]]:
    if args.input_dir:
        result_dir = Path(args.input_dir).expanduser().resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Input dir does not exist: {result_dir}")
        if not args.scene_dir:
            raise ValueError("--scene-dir is required when using --input-dir")
        scene_dir = Path(args.scene_dir).expanduser().resolve()
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene dir does not exist: {scene_dir}")
        yield result_dir, scene_dir
        return

    results_root = Path(args.results_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    scene_dirs = discover_scene_dirs(results_root)
    if not scene_dirs:
        raise RuntimeError(f"No dust3r_meta.json found under: {results_root}")

    for result_dir in scene_dirs:
        meta = load_scene_meta(result_dir)
        scene_rel = resolve_scene_rel(result_dir, meta, results_root)
        if args.only_scene and args.only_scene not in str(scene_rel):
            continue
        scene_dir = data_root / scene_rel
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Expected scene dir not found for {scene_rel}: {scene_dir}")
        yield result_dir, scene_dir


def summarize(results: Sequence[dict]) -> dict:
    summary = {"num_scenes": len(results), "scenes": list(results)}
    if results:
        rmses = np.asarray([scene["ate_rmse"] for scene in results], dtype=np.float64)
        summary["mean_ate_rmse"] = float(np.mean(rmses))
        summary["median_ate_rmse"] = float(np.median(rmses))
    return summary


def main() -> None:
    args = parse_args()
    per_scene = []
    for result_dir, scene_dir in iter_jobs(args):
        result = evaluate_scene(result_dir, scene_dir, args.waymo_source)
        per_scene.append(result)
        print(
            f"[ATE] {result['scene_rel']}: rmse={result['ate_rmse']:.6f} "
            f"(matches={result['num_matches']}, convention={result['convention']})"
        )

    if not per_scene:
        raise RuntimeError("No scene matched the current filters.")

    summary = summarize(per_scene)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
