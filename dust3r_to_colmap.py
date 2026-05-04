#!/usr/bin/env python3
"""
Convert DUSt3R scene outputs into a COLMAP-style sparse scene layout.

The exported scene can be consumed directly by the local 3DGS scripts because
it contains:

  <output_root>/<scene_rel>/
    images/
      *.png
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt

By default, this script exports the resized/cropped RGB tensors saved under
`per_view/*_rgb.npy`, so the images stay consistent with DUSt3R's predicted
intrinsics.

Examples:
  python dust3r_to_colmap.py \
    --results-root results_dust3r \
    --output-root outputs/dust3r_to_colmap

  python dust3r_to_colmap.py \
    --input-dir results_dust3r/DL3DV-2 \
    --output-dir scenes_3dgs/DL3DV-2 \
    --max-points 500000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DUSt3R outputs to COLMAP-style sparse scenes.")
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
        "--output-dir",
        type=str,
        default="",
        help="Output directory for a single-scene conversion.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(SCRIPT_DIR / "scenes_3dgs"),
        help="Root where converted scenes will be written in batch mode.",
    )
    parser.add_argument(
        "--only-scene",
        type=str,
        default="",
        help="Only convert scenes whose relative path contains this substring.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Optional confidence threshold applied on points3d_conf.npy before export.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional cap on exported sparse points. 0 keeps all points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when downsampling sparse points.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing converted scene directory.",
    )
    return parser.parse_args()


def discover_scene_dirs(results_root: Path) -> list[Path]:
    scene_dirs = sorted(meta.parent for meta in results_root.rglob("dust3r_meta.json"))
    return scene_dirs


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_scene_rel(result_dir: Path, meta: dict, results_root: Path | None) -> Path:
    scene_rel = meta.get("scene_rel")
    if scene_rel:
        return Path(scene_rel)
    if results_root is not None:
        return result_dir.relative_to(results_root)
    return Path(result_dir.name)


def load_scene_meta(result_dir: Path) -> dict:
    with (result_dir / "dust3r_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_names(meta: dict) -> list[str]:
    image_paths = meta.get("image_paths", [])
    if not image_paths:
        raise KeyError(f"dust3r_meta.json is missing image_paths: {meta}")
    return [Path(p).name for p in image_paths]


def resolve_per_view_rgb_path(result_dir: Path, idx: int, image_name: str) -> Path:
    per_view_dir = result_dir / "per_view"
    exact = per_view_dir / f"{idx:03d}_{Path(image_name).stem}_rgb.npy"
    if exact.is_file():
        return exact

    matches = sorted(per_view_dir.glob(f"{idx:03d}_*_rgb.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous RGB tensor for image index {idx} in {per_view_dir}")
    raise FileNotFoundError(f"Could not find per-view RGB tensor for {image_name} under {per_view_dir}")


def export_images(result_dir: Path, image_names: Sequence[str], images_dir: Path) -> list[tuple[str, int, int]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    exported: list[tuple[str, int, int]] = []
    for idx, image_name in enumerate(image_names):
        rgb_path = resolve_per_view_rgb_path(result_dir, idx, image_name)
        rgb = np.load(rgb_path)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Unexpected RGB tensor shape in {rgb_path}: {rgb.shape}")

        rgb_u8 = np.clip(rgb * 255.0, 0, 255).round().astype(np.uint8)
        out_path = images_dir / image_name
        Image.fromarray(rgb_u8).save(out_path)
        height, width = rgb_u8.shape[:2]
        exported.append((image_name, width, height))
    return exported


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    r_xx, r_yx, r_zx, r_xy, r_yy, r_zy, r_xz, r_yz, r_zz = rot.flat
    k = np.array(
        [
            [r_xx - r_yy - r_zz, 0.0, 0.0, 0.0],
            [r_yx + r_xy, r_yy - r_xx - r_zz, 0.0, 0.0],
            [r_zx + r_xz, r_zy + r_yz, r_zz - r_xx - r_yy, 0.0],
            [r_yz - r_zy, r_zx - r_xz, r_xy - r_yx, r_xx + r_yy + r_zz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_txt(path: Path, intrinsics: np.ndarray, image_sizes: Sequence[tuple[str, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(image_sizes)}\n")
        for idx, ((_, width, height), intrinsic) in enumerate(zip(image_sizes, intrinsics), start=1):
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            # This local gaussian-splatting fork assumes PINHOLE in its COLMAP loader.
            model = "PINHOLE"
            params = [fx, fy, cx, cy]
            params_str = " ".join(f"{value:.8f}" for value in params)
            f.write(f"{idx} {model} {width} {height} {params_str}\n")


def write_images_txt(path: Path, cam2world: np.ndarray, image_sizes: Sequence[tuple[str, int, int]]) -> None:
    world2cam = np.linalg.inv(cam2world)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_sizes)}, mean observations per image: 0\n\n")
        for idx, ((image_name, _, _), w2c) in enumerate(zip(image_sizes, world2cam), start=1):
            rot = w2c[:3, :3]
            tvec = w2c[:3, 3]
            qvec = rotmat_to_qvec(rot)
            q_str = " ".join(f"{value:.10f}" for value in qvec)
            t_str = " ".join(f"{value:.10f}" for value in tvec)
            f.write(f"{idx} {q_str} {t_str} {idx} {image_name}\n")
            f.write("\n")


def select_points(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    if conf_threshold > 0:
        mask &= conf >= conf_threshold

    points = points[mask]
    colors = colors[mask]
    conf = conf[mask]

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        indices.sort()
        points = points[indices]
        colors = colors[indices]
        conf = conf[indices]
    return points, colors, conf


def write_points3d_txt(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for idx, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            x, y, z = (float(v) for v in xyz)
            r, g, b = (int(v) for v in rgb)
            f.write(f"{idx} {x:.8f} {y:.8f} {z:.8f} {r} {g} {b} 0.0\n")


def write_points3d_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            x, y, z = (float(v) for v in xyz)
            r, g, b = (int(v) for v in rgb)
            f.write(f"{x:.8f} {y:.8f} {z:.8f} 0.0 0.0 0.0 {r} {g} {b}\n")


def convert_scene(
    result_dir: Path,
    output_dir: Path,
    scene_rel: Path,
    conf_threshold: float,
    max_points: int,
    seed: int,
    overwrite: bool,
) -> dict:
    meta = load_scene_meta(result_dir)
    image_names = load_image_names(meta)

    cam2world = np.load(result_dir / "cam2world.npy")
    intrinsics = np.load(result_dir / "intrinsics.npy")
    points = np.load(result_dir / "points3d_world.npy")
    colors = np.load(result_dir / "points3d_rgb.npy")
    conf = np.load(result_dir / "points3d_conf.npy")

    if cam2world.shape[0] != len(image_names) or intrinsics.shape[0] != len(image_names):
        raise ValueError(
            f"Shape mismatch in {result_dir}: num_images={len(image_names)}, "
            f"cam2world={cam2world.shape}, intrinsics={intrinsics.shape}"
        )

    ensure_empty_dir(output_dir, overwrite=overwrite)
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_sizes = export_images(result_dir, image_names, images_dir)
    points, colors, conf = select_points(points, colors, conf, conf_threshold, max_points, seed)

    write_cameras_txt(sparse_dir / "cameras.txt", intrinsics, image_sizes)
    write_images_txt(sparse_dir / "images.txt", cam2world, image_sizes)
    write_points3d_txt(sparse_dir / "points3D.txt", points, colors)
    write_points3d_ply(sparse_dir / "points3D.ply", points, colors)

    export_meta = {
        "scene_rel": str(scene_rel),
        "result_dir": str(result_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_images": len(image_names),
        "num_points": int(len(points)),
        "conf_threshold": float(conf_threshold),
        "max_points": int(max_points),
        "source_model": meta.get("model", ""),
        "image_source": "per_view_rgb",
    }
    with (output_dir / "dust3r_to_colmap_meta.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)
    return export_meta


def iter_jobs(args: argparse.Namespace) -> Iterable[tuple[Path, Path, Path]]:
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
        meta = load_scene_meta(input_dir)
        scene_rel = resolve_scene_rel(input_dir, meta, results_root=None)
        if not args.output_dir:
            raise ValueError("--output-dir is required when using --input-dir")
        output_dir = Path(args.output_dir).expanduser().resolve()
        yield input_dir, output_dir, scene_rel
        return

    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    scene_dirs = discover_scene_dirs(results_root)
    if not scene_dirs:
        raise RuntimeError(f"No dust3r_meta.json found under: {results_root}")

    for result_dir in scene_dirs:
        meta = load_scene_meta(result_dir)
        scene_rel = resolve_scene_rel(result_dir, meta, results_root=results_root)
        if args.only_scene and args.only_scene not in str(scene_rel):
            continue
        output_dir = output_root / scene_rel
        yield result_dir, output_dir, scene_rel


def main() -> None:
    args = parse_args()
    converted = []
    for result_dir, output_dir, scene_rel in iter_jobs(args):
        print(f"[convert] {scene_rel} -> {output_dir}")
        export_meta = convert_scene(
            result_dir=result_dir,
            output_dir=output_dir,
            scene_rel=scene_rel,
            conf_threshold=args.conf_threshold,
            max_points=args.max_points,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        converted.append(export_meta)
        print(
            f"  exported {export_meta['num_images']} images and "
            f"{export_meta['num_points']} sparse points"
        )

    if not converted:
        raise RuntimeError("No scene matched the current filters.")

    print(f"\nConverted {len(converted)} scene(s).")


if __name__ == "__main__":
    main()
