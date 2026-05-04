from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .config import (
    ProjectConfig,
    get_scene_spec,
    make_run_id,
    read_json,
    scene_run_dir,
    write_json,
)


_COLMAP_IO: ModuleType | None = None


@dataclass(frozen=True)
class ResolvedScene:
    scene_key: str
    scene_rel: Path
    base_scene_dir: Path
    source_type: str


def _load_colmap_io(project_root: Path) -> ModuleType:
    global _COLMAP_IO
    if _COLMAP_IO is not None:
        return _COLMAP_IO

    module_path = project_root / "gaussian-splatting" / "utils" / "read_write_model.py"
    spec = importlib.util.spec_from_file_location("gs_read_write_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _COLMAP_IO = module
    return module


def _dust3r_variant_root(config: ProjectConfig, variant: str) -> Path:
    root = config.dust3r_to_colmap_root.expanduser()
    if root.name == variant:
        return root
    if root.name.startswith("dust3r_"):
        return root.parent / variant
    return root / variant


def resolve_base_scene(config: ProjectConfig, scene_name: str, dust3r_variant: str | None = None) -> ResolvedScene:
    scene = get_scene_spec(scene_name)
    variant = dust3r_variant or str(config.defaults.get("dust3r_variant", "dust3r_light"))
    fallback_variants = [variant] + [item for item in ("dust3r_light", "dust3r_heavy", "dust3r_none") if item != variant]

    candidates: list[tuple[str, Path]] = [
        (f"dust3r_to_colmap/{item}", _dust3r_variant_root(config, item) / scene.scene_rel)
        for item in fallback_variants
    ]

    for source_type, candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "sparse" / "0").is_dir():
            return ResolvedScene(
                scene_key=scene.key,
                scene_rel=scene.scene_rel,
                base_scene_dir=candidate.expanduser(),
                source_type=source_type,
            )

    raise FileNotFoundError(
        "No reusable Part 2 scene was found. "
        f"Tried: {[str(path) for _, path in candidates]}"
    )


def suggest_scene_build_commands(config: ProjectConfig, scene_name: str, dust3r_variant: str | None = None) -> list[str]:
    scene = get_scene_spec(scene_name)
    variant = dust3r_variant or str(config.defaults.get("dust3r_variant", "dust3r_light"))
    datasets_key = "405841" if scene.key == "405841_FRONT" else scene.key
    output_dir = _dust3r_variant_root(config, variant) / scene.scene_rel
    return [
        f"python3 {config.project_root / 'subsample_p2_frames.py'} "
        f"--data_root {config.project_root / 'data'} "
        f"--out_root {config.data_p2_sparse_root} --datasets {datasets_key}",
        f"python3 {config.project_root / 'run_dust3r_inference.py'} "
        f"--root {config.data_p2_sparse_root} "
        f"--output-root {config.dust3r_outputs_root / ('results_' + variant)} "
        f"--only-scene '{scene.scene_rel}'",
        f"python3 {config.project_root / 'dust3r_to_colmap.py'} "
        f"--input-dir {config.dust3r_outputs_root / ('results_' + variant) / scene.scene_rel} "
        f"--output-dir {output_dir} --overwrite",
    ]


def _camera_center_from_w2c(rot_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    return -rot_cw.T @ t_cw


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(max(min(dot, 1.0), -1.0))
    sin_theta_0 = math.sin(theta_0)
    theta_t = theta_0 * t
    s0 = math.sin(theta_0 - theta_t) / sin_theta_0
    s1 = math.sin(theta_t) / sin_theta_0
    return s0 * q0 + s1 * q1


def _image_to_pose_bundle(colmap_io: ModuleType, image: Any) -> dict[str, Any]:
    rot_cw = colmap_io.qvec2rotmat(image.qvec)
    center = _camera_center_from_w2c(rot_cw, image.tvec)
    rot_wc = rot_cw.T
    return {
        "qvec": np.asarray(image.qvec, dtype=np.float64),
        "tvec": np.asarray(image.tvec, dtype=np.float64),
        "rot_wc": rot_wc,
        "center": center,
    }


def _pair_candidates(sorted_images: list[Any], max_pairs: int) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for first, second in zip(sorted_images[:-1], sorted_images[1:]):
        pairs.append((first, second))
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    return pairs


def build_trajectory_manifest(
    config: ProjectConfig,
    scene_name: str,
    base_scene_dir: str | Path | None = None,
    num_intermediate_views: int | None = None,
    max_pairs: int | None = None,
    dust3r_variant: str | None = None,
    run_id: str | None = None,
) -> Path:
    resolved = resolve_base_scene(config, scene_name, dust3r_variant) if base_scene_dir is None else ResolvedScene(
        scene_key=get_scene_spec(scene_name).key,
        scene_rel=get_scene_spec(scene_name).scene_rel,
        base_scene_dir=Path(base_scene_dir).expanduser(),
        source_type="explicit_path",
    )
    num_views = int(num_intermediate_views or config.defaults.get("num_intermediate_views", 4))
    pair_limit = int(config.defaults.get("max_pairs", 2) if max_pairs is None else max_pairs)
    run_name = run_id or make_run_id(resolved.scene_key)
    run_dir = scene_run_dir(config, resolved.scene_key, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    colmap_io = _load_colmap_io(config.project_root)
    cameras, images, _points3d = colmap_io.read_model(str(resolved.base_scene_dir / "sparse" / "0"), ext="")
    ordered_images = sorted(images.values(), key=lambda item: item.name)
    pairs = _pair_candidates(ordered_images, pair_limit)

    manifest_pairs: list[dict[str, Any]] = []
    for pair_idx, (img_a, img_b) in enumerate(pairs):
        pose_a = _image_to_pose_bundle(colmap_io, img_a)
        pose_b = _image_to_pose_bundle(colmap_io, img_b)
        camera = cameras[img_a.camera_id]
        start_camera = cameras[img_a.camera_id]
        end_camera = cameras[img_b.camera_id]

        views: list[dict[str, Any]] = []
        for view_idx in range(num_views):
            alpha = (view_idx + 1) / (num_views + 1)
            center = (1.0 - alpha) * pose_a["center"] + alpha * pose_b["center"]
            rot_wc = colmap_io.qvec2rotmat(_slerp(pose_a["qvec"], pose_b["qvec"], alpha)).T
            rot_cw = rot_wc.T
            tvec = -rot_cw @ center
            qvec = colmap_io.rotmat2qvec(rot_cw)
            views.append(
                {
                    "index": view_idx,
                    "alpha": alpha,
                    "qvec": qvec.tolist(),
                    "tvec": tvec.tolist(),
                    "output_name": f"pseudo_pair_{pair_idx:04d}_{view_idx:02d}.png",
                    "camera_id": int(img_a.camera_id),
                    "camera_model": camera.model,
                    "camera_params": [float(v) for v in np.asarray(camera.params)],
                    "width": int(camera.width),
                    "height": int(camera.height),
                }
            )

        manifest_pairs.append(
            {
                "pair_id": f"pair_{pair_idx:04d}",
                "start_image": img_a.name,
                "end_image": img_b.name,
                "start_image_path": str((resolved.base_scene_dir / "images" / img_a.name).expanduser()),
                "end_image_path": str((resolved.base_scene_dir / "images" / img_b.name).expanduser()),
                "start_qvec": pose_a["qvec"].tolist(),
                "start_tvec": pose_a["tvec"].tolist(),
                "start_camera_id": int(img_a.camera_id),
                "start_camera_model": start_camera.model,
                "start_camera_params": [float(v) for v in np.asarray(start_camera.params)],
                "start_width": int(start_camera.width),
                "start_height": int(start_camera.height),
                "end_qvec": pose_b["qvec"].tolist(),
                "end_tvec": pose_b["tvec"].tolist(),
                "end_camera_id": int(img_b.camera_id),
                "end_camera_model": end_camera.model,
                "end_camera_params": [float(v) for v in np.asarray(end_camera.params)],
                "end_width": int(end_camera.width),
                "end_height": int(end_camera.height),
                "camera_id": int(img_a.camera_id),
                "camera_model": camera.model,
                "camera_params": [float(v) for v in np.asarray(camera.params)],
                "width": int(camera.width),
                "height": int(camera.height),
                "intermediate_views": views,
            }
        )

    payload = {
        "scene": resolved.scene_key,
        "scene_rel": str(resolved.scene_rel),
        "source_type": resolved.source_type,
        "base_scene_dir": str(resolved.base_scene_dir),
        "run_id": run_name,
        "run_dir": str(run_dir),
        "num_intermediate_views": num_views,
        "max_pairs": pair_limit,
        "num_anchor_pairs": len(manifest_pairs),
        "pairs": manifest_pairs,
    }
    return write_json(run_dir / "trajectory_manifest.json", payload)


def load_trajectory_bundle(path: str | Path) -> dict[str, Any]:
    return read_json(path)
