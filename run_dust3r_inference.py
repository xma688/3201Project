#!/usr/bin/env python3
"""
Run DUSt3R inference on sparse-view scenes under a project root.

This script reuses the pair generation logic from `build_pairs.py`, writes the
pair manifests if requested, then feeds those pairs directly into DUSt3R
inference and global alignment.

Outputs per scene:
  <output_root>/<scene_rel>/
    dust3r_meta.json
    pairs_used.json
    pair_indices.npy
    cam2world.npy
    world2cam.npy
    intrinsics.npy
    focals.npy
    principal_points.npy
    points3d_world.npy
    points3d_rgb.npy
    points3d_conf.npy
    points3d_view_ids.npy
    per_view/
      *_pts3d.npy
      *_depth.npy
      *_conf.npy
      *_mask.npy
      *_rgb.npy
    points.ply

python run_dust3r_inference.py --root data_p2_sparse --output-root results_dust3r
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from build_pairs import (
    IMAGE_DIRNAMES_DEFAULT,
    IMAGE_EXTS_DEFAULT,
    build_pair_indices,
    discover_image_dirs,
    list_images,
    make_pair_records,
    save_pairs_for_scene,
)


SCRIPT_DIR = Path(__file__).resolve().parent
np = None
torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DUSt3R inference on sparse-view project scenes.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(SCRIPT_DIR / "data_p2_sparse"),
        help="Root directory containing sparse-view scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(SCRIPT_DIR / "results_dust3r"),
        help="Directory where DUSt3R outputs will be saved.",
    )
    parser.add_argument(
        "--dust3r-repo",
        type=str,
        default=str(SCRIPT_DIR / "dust3r"),
        help="Path to the local DUSt3R repository clone.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional local checkpoint path. If omitted, prefer the local DUSt3R .pth if present.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        help="Fallback model identifier when no local checkpoint is provided.",
    )
    parser.add_argument(
        "--only-scene",
        type=str,
        default="",
        help="Only process scenes whose relative path contains this substring.",
    )
    parser.add_argument(
        "--image-dirnames",
        type=str,
        nargs="+",
        default=IMAGE_DIRNAMES_DEFAULT,
        help="Folder names to search for, e.g. rgb images",
    )
    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=IMAGE_EXTS_DEFAULT,
        help="Supported image extensions for scene discovery.",
    )
    parser.add_argument(
        "--pairs-dirname",
        type=str,
        default="pairs",
        help="Directory name used to save pair manifests under each scene.",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default="swin-2",
        help="DUSt3R scene graph, e.g. complete, swin-2, logwin-3-noncyclic, oneref-0",
    )
    parser.add_argument(
        "--prefilter",
        type=str,
        default="none",
        help="Optional filter applied after pair generation: none, seqN, cycN.",
    )
    parser.add_argument(
        "--symmetrize",
        dest="symmetrize",
        action="store_true",
        help="Use symmetrized DUSt3R inference pairs.",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help="Disable symmetric reversed pairs.",
    )
    parser.set_defaults(symmetrize=True)
    parser.add_argument(
        "--write-pairs-manifest",
        dest="write_pairs_manifest",
        action="store_true",
        help="Write pairs.json / pairs.txt / pairs_meta.json under each scene.",
    )
    parser.add_argument(
        "--no-write-pairs-manifest",
        dest="write_pairs_manifest",
        action="store_false",
        help="Skip writing pair manifests and only keep in-memory pairs.",
    )
    parser.set_defaults(write_pairs_manifest=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[224, 512],
        help="DUSt3R input image size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. DUSt3R will force 1 if image shapes differ.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Global alignment LR schedule.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=300,
        help="Number of global alignment iterations for multi-view scenes.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Global alignment learning rate.",
    )
    parser.add_argument(
        "--min-conf-thr",
        type=float,
        default=3.0,
        help="DUSt3R visualization-style confidence threshold used for exported masks / point cloud.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, cpu, cuda:0, ...",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes that already contain dust3r_meta.json in the output directory.",
    )
    parser.add_argument(
        "--export-ply",
        dest="export_ply",
        action="store_true",
        help="Export a filtered fused point cloud as points.ply.",
    )
    parser.add_argument(
        "--no-export-ply",
        dest="export_ply",
        action="store_false",
        help="Disable PLY export.",
    )
    parser.set_defaults(export_ply=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce DUSt3R console output.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_weights(weights_arg: str, model_name: str, dust3r_repo: Path) -> str:
    if weights_arg:
        return str(Path(weights_arg).expanduser().resolve())

    local_ckpt = dust3r_repo / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    if local_ckpt.is_file():
        return str(local_ckpt)

    return model_name


def import_dust3r_modules(dust3r_repo: Path):
    repo_str = str(dust3r_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    return {
        "AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo,
        "GlobalAlignerMode": GlobalAlignerMode,
        "global_aligner": global_aligner,
        "inference": inference,
        "load_images": load_images,
        "to_numpy": to_numpy,
    }


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def build_intrinsics_from_scene(scene, to_numpy_fn):
    intrinsics = to_numpy_fn(scene.get_intrinsics())
    focals = np.asarray(to_numpy_fn(scene.get_focals()))
    principal_points = np.asarray(to_numpy_fn(scene.get_principal_points()))
    return intrinsics, focals, principal_points


def save_scene_outputs(
    out_dir: Path,
    scene,
    to_numpy_fn,
    images: Sequence[Path],
    pair_indices: Sequence[tuple[int, int]],
    pair_records: Sequence[dict],
    meta: dict,
    export_ply: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view_dir = out_dir / "per_view"
    per_view_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, focals, principal_points = build_intrinsics_from_scene(scene, to_numpy_fn)
    cam2world = np.asarray(to_numpy_fn(scene.get_im_poses()))
    world2cam = np.linalg.inv(cam2world)
    depths = to_numpy_fn(scene.get_depthmaps())
    pts3d = to_numpy_fn(scene.get_pts3d())
    confs = to_numpy_fn([c for c in scene.im_conf])
    masks = to_numpy_fn(scene.get_masks())
    rgb_imgs = to_numpy_fn(scene.imgs) if scene.imgs is not None else None

    np.save(out_dir / "cam2world.npy", cam2world)
    np.save(out_dir / "world2cam.npy", world2cam)
    np.save(out_dir / "intrinsics.npy", intrinsics)
    np.save(out_dir / "focals.npy", focals)
    np.save(out_dir / "principal_points.npy", principal_points)
    np.save(out_dir / "pair_indices.npy", np.asarray(pair_indices, dtype=np.int32))

    all_points = []
    all_colors = []
    all_conf = []
    all_view_ids = []

    for idx, image_path in enumerate(images):
        prefix = f"{idx:03d}_{image_path.stem}"
        np.save(per_view_dir / f"{prefix}_pts3d.npy", np.asarray(pts3d[idx]))
        np.save(per_view_dir / f"{prefix}_depth.npy", np.asarray(depths[idx]))
        np.save(per_view_dir / f"{prefix}_conf.npy", np.asarray(confs[idx]))
        np.save(per_view_dir / f"{prefix}_mask.npy", np.asarray(masks[idx], dtype=bool))
        if rgb_imgs is not None:
            np.save(per_view_dir / f"{prefix}_rgb.npy", np.asarray(rgb_imgs[idx]))

        mask = np.asarray(masks[idx], dtype=bool)
        if not np.any(mask):
            continue

        points = np.asarray(pts3d[idx])[mask].reshape(-1, 3)
        conf = np.asarray(confs[idx])[mask].reshape(-1)

        if rgb_imgs is not None:
            colors = np.clip(np.asarray(rgb_imgs[idx])[mask], 0, 1)
            colors = (colors * 255.0).round().astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)

        all_points.append(points)
        all_colors.append(colors)
        all_conf.append(conf)
        all_view_ids.append(np.full(len(points), idx, dtype=np.int32))

    if all_points:
        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0)
        fused_conf = np.concatenate(all_conf, axis=0)
        fused_view_ids = np.concatenate(all_view_ids, axis=0)
    else:
        fused_points = np.zeros((0, 3), dtype=np.float32)
        fused_colors = np.zeros((0, 3), dtype=np.uint8)
        fused_conf = np.zeros((0,), dtype=np.float32)
        fused_view_ids = np.zeros((0,), dtype=np.int32)

    np.save(out_dir / "points3d_world.npy", fused_points)
    np.save(out_dir / "points3d_rgb.npy", fused_colors)
    np.save(out_dir / "points3d_conf.npy", fused_conf)
    np.save(out_dir / "points3d_view_ids.npy", fused_view_ids)

    if export_ply and len(fused_points) > 0:
        write_ascii_ply(out_dir / "points.ply", fused_points, fused_colors)

    with (out_dir / "pairs_used.json").open("w", encoding="utf-8") as f:
        json.dump(list(pair_records), f, indent=2, ensure_ascii=False)

    export_meta = dict(meta)
    export_meta.update(
        {
            "num_exported_points": int(len(fused_points)),
            "output_dir": str(out_dir.resolve()),
            "per_view_dir": str(per_view_dir.resolve()),
            "cam2world_shape": list(cam2world.shape),
            "intrinsics_shape": list(intrinsics.shape),
            "focals_shape": list(np.asarray(focals).shape),
        }
    )
    with (out_dir / "dust3r_meta.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)

    return export_meta


def main() -> None:
    args = parse_args()

    global np, torch
    import numpy as np  # type: ignore[no-redef]
    import torch  # type: ignore[no-redef]
    root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dust3r_repo = Path(args.dust3r_repo).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not dust3r_repo.exists():
        raise FileNotFoundError(f"DUSt3R repo does not exist: {dust3r_repo}")

    image_dirs = discover_image_dirs(root, args.image_dirnames, args.exts)
    if not image_dirs:
        raise RuntimeError(
            f"No image folders named {args.image_dirnames} with supported files found under: {root}"
        )

    device = resolve_device(args.device)
    dust3r_mod = import_dust3r_modules(dust3r_repo)
    AsymmetricCroCo3DStereo = dust3r_mod["AsymmetricCroCo3DStereo"]
    GlobalAlignerMode = dust3r_mod["GlobalAlignerMode"]
    global_aligner = dust3r_mod["global_aligner"]
    inference = dust3r_mod["inference"]
    load_images = dust3r_mod["load_images"]
    to_numpy_fn = dust3r_mod["to_numpy"]

    weights = resolve_weights(args.weights, args.model_name, dust3r_repo)
    if not args.quiet:
        print(f"Using device : {device}")
        print(f"Using weights: {weights}")

    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)
    model.eval()
    square_ok = bool(getattr(model, "square_ok", False))
    patch_size = int(getattr(model, "patch_size", 16))

    processed = 0
    skipped = 0

    for image_dir in image_dirs:
        scene_dir = image_dir.parent
        scene_rel = scene_dir.relative_to(root)
        scene_rel_str = str(scene_rel)

        if args.only_scene and args.only_scene not in scene_rel_str:
            skipped += 1
            continue

        out_dir = output_root / scene_rel
        if args.skip_existing and (out_dir / "dust3r_meta.json").is_file():
            print(f"[skip] {scene_rel_str}: output already exists")
            skipped += 1
            continue

        images = list_images(image_dir, args.exts)
        if len(images) < 2:
            print(f"[skip] {scene_rel_str}: fewer than 2 images")
            skipped += 1
            continue

        pair_indices = build_pair_indices(
            num_images=len(images),
            scene_graph=args.scene_graph,
            prefilter=args.prefilter,
            symmetrize=args.symmetrize,
        )
        if not pair_indices:
            print(f"[skip] {scene_rel_str}: no pairs generated")
            skipped += 1
            continue

        if args.write_pairs_manifest:
            pair_records, pairs_meta = save_pairs_for_scene(
                root=root,
                image_dir=image_dir,
                images=images,
                pairs=pair_indices,
                output_dirname=args.pairs_dirname,
                scene_graph=args.scene_graph,
                prefilter=args.prefilter,
                symmetrize=args.symmetrize,
                dry_run=False,
            )
        else:
            pair_records, pairs_meta = make_pair_records(
                root=root,
                image_dir=image_dir,
                images=images,
                pairs=pair_indices,
                scene_graph=args.scene_graph,
                prefilter=args.prefilter,
                symmetrize=args.symmetrize,
            )

        if not args.quiet:
            print(f"[run] {scene_rel_str}")
            print(f"  images : {len(images)}")
            print(f"  pairs  : {len(pair_indices)}")

        loaded_imgs = load_images(
            [str(p) for p in images],
            size=args.image_size,
            verbose=not args.quiet,
            patch_size=patch_size,
            square_ok=square_ok,
        )
        if len(loaded_imgs) != len(images):
            raise RuntimeError(
                f"Loaded image count mismatch for {scene_rel_str}: "
                f"expected {len(images)}, got {len(loaded_imgs)}"
            )

        dust3r_pairs = [(loaded_imgs[i], loaded_imgs[j]) for i, j in pair_indices]
        output = inference(
            dust3r_pairs,
            model,
            device,
            batch_size=args.batch_size,
            verbose=not args.quiet,
        )

        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(loaded_imgs) > 2
            else GlobalAlignerMode.PairViewer
        )
        scene = global_aligner(output, device=device, mode=mode, verbose=not args.quiet)

        alignment_loss = None
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            alignment_loss = scene.compute_global_alignment(
                init="mst",
                niter=args.niter,
                schedule=args.schedule,
                lr=args.lr,
            )
            alignment_loss = float(alignment_loss)

        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))

        export_meta = {
            **pairs_meta,
            "model": weights,
            "device": device,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "global_alignment_mode": mode.value,
            "schedule": args.schedule,
            "niter": args.niter,
            "lr": args.lr,
            "user_min_conf_thr": args.min_conf_thr,
            "internal_min_conf_thr": float(scene.min_conf_thr),
            "alignment_loss": alignment_loss,
            "image_paths": [str(p.resolve()) for p in images],
        }

        save_scene_outputs(
            out_dir=out_dir,
            scene=scene,
            to_numpy_fn=to_numpy_fn,
            images=images,
            pair_indices=pair_indices,
            pair_records=pair_records,
            meta=export_meta,
            export_ply=args.export_ply,
        )
        processed += 1

    print("\nDone.")
    print(f"Processed scenes: {processed}")
    print(f"Skipped scenes  : {skipped}")


if __name__ == "__main__":
    main()
