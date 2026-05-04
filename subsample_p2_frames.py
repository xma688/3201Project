#!/usr/bin/env python3
"""
Sub-sample Project 4 datasets for Part 2 sparse-view reconstruction.

Default behavior follows the project instruction:
- Waymo-405841: keep 1 / 10 frames
- DL3DV-2:      keep 1 / 30 frames
- Re10k-1:      keep 1 / 30 frames

Expected input layout (matching the user's current data tree):

<data_root>/
  405841/FRONT/{rgb, calib, depth, gt, gt_old}
  DL3DV-2/{rgb, cameras.json, intrinsics.json}
  Re10k-1/{images, cameras.json, intrinsics.json}

Output layout:

<out_root>/
  405841/FRONT/rgb/
  405841/FRONT/meta/          # optional non-pose metadata (e.g. calib)
  405841/FRONT/eval_meta/     # optional GT / pose-related files for evaluation only
  DL3DV-2/rgb/
  DL3DV-2/intrinsics.json
  DL3DV-2/eval_meta/cameras.json
  Re10k-1/images/
  Re10k-1/intrinsics.json
  Re10k-1/eval_meta/cameras.json

Important:
- To match the Part 2 requirement of "NO camera poses provided", this script does NOT
  place camera pose files into the main sparse input folder by default.
- If you still need GT poses later for ATE evaluation, use --save-eval-meta.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "405841": {
        "scene_rel": Path("405841/FRONT"),
        "image_dir": "rgb",
        "step": 10,
        "copy_to_meta": ["calib"],
        "filter_to_eval": ["depth", "gt", "gt_old"],
        "copy_files": [],
        "filter_json_to_eval": [],
    },
    "DL3DV-2": {
        "scene_rel": Path("DL3DV-2"),
        "image_dir": "rgb",
        "step": 30,
        "copy_to_meta": [],
        "filter_to_eval": [],
        "copy_files": ["intrinsics.json"],
        "filter_json_to_eval": ["cameras.json"],
    },
    "Re10k-1": {
        "scene_rel": Path("Re10k-1"),
        "image_dir": "images",
        "step": 30,
        "copy_to_meta": [],
        "filter_to_eval": [],
        "copy_files": ["intrinsics.json"],
        "filter_json_to_eval": ["cameras.json"],
    },
}


def natural_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]



def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        raise RuntimeError(f"No images found in: {folder}")
    return files



def choose_indices(n: int, step: int, start: int = 0, keep_last: bool = True) -> List[int]:
    if n <= 0:
        return []
    idx = list(range(start, n, step))
    if not idx:
        idx = [0]
    if keep_last and idx[-1] != n - 1:
        idx.append(n - 1)
    return sorted(set(idx))



def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def copy_selected_files(files: Sequence[Path], dst_dir: Path) -> None:
    safe_mkdir(dst_dir)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)



def copy_or_filter_dir(src_dir: Path, dst_dir: Path, selected_names: Set[str], selected_stems: Set[str]) -> Tuple[int, int]:
    """Copy files from src_dir to dst_dir.

    Strategy:
    - If filenames appear to correspond to selected frames, copy only matched items.
    - Otherwise, copy the whole directory as-is.

    Returns:
        (copied_count, total_count)
    """
    if not src_dir.exists():
        return 0, 0

    items = [p for p in src_dir.iterdir() if p.is_file()]
    total = len(items)
    if total == 0:
        safe_mkdir(dst_dir)
        return 0, 0

    matched = []
    for p in items:
        if p.name in selected_names or p.stem in selected_stems:
            matched.append(p)

    safe_mkdir(dst_dir)
    if matched:
        for p in matched:
            shutil.copy2(p, dst_dir / p.name)
        return len(matched), total

    # Fallback: filenames do not line up with frame names; copy everything.
    for p in items:
        shutil.copy2(p, dst_dir / p.name)
    return total, total



def filter_list_by_names(items: Sequence[Any], selected_names: Set[str], selected_stems: Set[str]) -> List[Any]:
    filtered = []
    for item in items:
        if isinstance(item, dict):
            hit = False
            for v in item.values():
                if isinstance(v, str):
                    name = Path(v).name
                    stem = Path(v).stem
                    if name in selected_names or stem in selected_stems:
                        hit = True
                        break
            if hit:
                filtered.append(item)
        elif isinstance(item, str):
            name = Path(item).name
            stem = Path(item).stem
            if name in selected_names or stem in selected_stems:
                filtered.append(item)
    return filtered



def filter_json_content(obj: Any, selected_names: Set[str], selected_stems: Set[str]) -> Tuple[Any, bool]:
    """Best-effort JSON filtering.

    Returns:
        (filtered_object, changed)
    """
    # Case 1: top-level dict keyed by image name / stem.
    if isinstance(obj, dict):
        keys = list(obj.keys())
        keyed_by_name = sum((k in selected_names or Path(k).stem in selected_stems) for k in keys)
        if keyed_by_name > 0:
            filtered = {k: v for k, v in obj.items() if (k in selected_names or Path(k).stem in selected_stems)}
            return filtered, True

        # Case 2: common nested containers.
        changed = False
        new_obj: Dict[str, Any] = dict(obj)
        for key in ["frames", "images", "cameras", "poses", "records", "data"]:
            if key in obj and isinstance(obj[key], list):
                filtered_list = filter_list_by_names(obj[key], selected_names, selected_stems)
                if filtered_list:
                    new_obj[key] = filtered_list
                    changed = True
        return new_obj, changed

    # Case 3: top-level list.
    if isinstance(obj, list):
        filtered = filter_list_by_names(obj, selected_names, selected_stems)
        if filtered:
            return filtered, True

    return obj, False



def filter_json_file(src_json: Path, dst_json: Path, selected_names: Set[str], selected_stems: Set[str]) -> str:
    safe_mkdir(dst_json.parent)
    try:
        with src_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        filtered, changed = filter_json_content(data, selected_names, selected_stems)
        with dst_json.open("w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        if changed:
            return "filtered"
        return "copied_unfiltered_structure_unknown"
    except Exception as e:
        shutil.copy2(src_json, dst_json)
        return f"copied_raw_due_to_parse_error: {e}"



def write_manifest(
    out_scene: Path,
    dataset_name: str,
    img_dir_name: str,
    selected_files: Sequence[Path],
    step: int,
    keep_last: bool,
) -> None:
    selected_names = [p.name for p in selected_files]
    manifest = {
        "dataset": dataset_name,
        "image_dir": img_dir_name,
        "step": step,
        "keep_last": keep_last,
        "num_selected": len(selected_names),
        "selected_frames": selected_names,
    }
    with (out_scene / "subsample_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with (out_scene / "selected_frames.txt").open("w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(name + "\n")



def process_dataset(
    data_root: Path,
    out_root: Path,
    dataset_name: str,
    cfg: Dict[str, Any],
    keep_last: bool,
    start: int,
    save_eval_meta: bool,
    dry_run: bool,
) -> None:
    scene_in = data_root / cfg["scene_rel"]
    scene_out = out_root / cfg["scene_rel"]
    img_in = scene_in / cfg["image_dir"]
    img_out = scene_out / cfg["image_dir"]

    all_imgs = list_images(img_in)
    indices = choose_indices(len(all_imgs), cfg["step"], start=start, keep_last=keep_last)
    selected = [all_imgs[i] for i in indices]
    selected_names = {p.name for p in selected}
    selected_stems = {p.stem for p in selected}

    print(f"\n[{dataset_name}]")
    print(f"  input scene : {scene_in}")
    print(f"  image dir   : {img_in}")
    print(f"  total imgs  : {len(all_imgs)}")
    print(f"  step        : 1/{cfg['step']}")
    print(f"  selected    : {len(selected)}")
    print(f"  output scene: {scene_out}")

    if dry_run:
        return

    safe_mkdir(scene_out)
    copy_selected_files(selected, img_out)
    write_manifest(scene_out, dataset_name, cfg["image_dir"], selected, cfg["step"], keep_last)

    # Non-pose metadata that can stay with the sparse set.
    for folder_name in cfg["copy_to_meta"]:
        src_dir = scene_in / folder_name
        dst_dir = scene_out / "meta" / folder_name
        copied, total = copy_or_filter_dir(src_dir, dst_dir, selected_names, selected_stems)
        print(f"  meta dir    : {folder_name} -> copied {copied}/{total}")

    # Files to copy directly to sparse scene root (e.g. intrinsics.json).
    for file_name in cfg["copy_files"]:
        src = scene_in / file_name
        if src.exists():
            shutil.copy2(src, scene_out / file_name)
            print(f"  copy file   : {file_name}")

    # Evaluation-only files.
    if save_eval_meta:
        for folder_name in cfg["filter_to_eval"]:
            src_dir = scene_in / folder_name
            dst_dir = scene_out / "eval_meta" / folder_name
            copied, total = copy_or_filter_dir(src_dir, dst_dir, selected_names, selected_stems)
            print(f"  eval dir    : {folder_name} -> copied {copied}/{total}")

        for json_name in cfg["filter_json_to_eval"]:
            src_json = scene_in / json_name
            if src_json.exists():
                dst_json = scene_out / "eval_meta" / json_name
                status = filter_json_file(src_json, dst_json, selected_names, selected_stems)
                print(f"  eval json   : {json_name} -> {status}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-sample Project 4 datasets for Part 2 sparse-view experiments.")
    parser.add_argument("--data_root", type=Path, required=True, help="Root folder containing 405841/, DL3DV-2/, Re10k-1/")
    parser.add_argument("--out_root", type=Path, required=True, help="Output root for sparse datasets")
    parser.add_argument("--datasets", nargs="+", default=["405841", "DL3DV-2", "Re10k-1"], choices=list(DATASETS.keys()))
    parser.add_argument("--start", type=int, default=0, help="Start index for sub-sampling, default: 0")
    parser.add_argument("--no_keep_last", action="store_true", help="Do not force keeping the last frame")
    parser.add_argument("--save_eval_meta", action="store_true", help="Save filtered GT / pose-related metadata for evaluation")
    parser.add_argument("--dry_run", action="store_true", help="Only print statistics, do not copy files")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    keep_last = not args.no_keep_last

    for dataset_name in args.datasets:
        process_dataset(
            data_root=args.data_root,
            out_root=args.out_root,
            dataset_name=dataset_name,
            cfg=DATASETS[dataset_name],
            keep_last=keep_last,
            start=args.start,
            save_eval_meta=args.save_eval_meta,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
