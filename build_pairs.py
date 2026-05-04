#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build explicit image pairs for sparse-view DUSt3R inference.

This script scans a root directory for scene image folders such as `rgb/` or
`images/`, then writes pair manifests per scene.

The pair-generation semantics are aligned with DUSt3R's official
`dust3r.image_pairs.make_pairs(...)` implementation:
- `complete`
- `swin[-K][-noncyclic]`
- `logwin[-K][-noncyclic]`
- `oneref[-I]`
- `prefilter=seqN|cycN`
- `symmetrize=True|False`

Example structure:
data_p2_sparse/
  405841/FRONT/rgb/*.png
  405841/FRONT/selected_frames.txt
  405841/FRONT/subsample_manifest.json
  DL3DV-2/rgb/*.png
  Re10k-1/images/*.png

Outputs (for each scene):
  <scene_dir>/pairs/pairs.json
  <scene_dir>/pairs/pairs.txt
  <scene_dir>/pairs/pairs_meta.json

usage:
python build_pairs.py \
  --root /path/to/your/CVproj/data_p2_sparse \
  --only-scene 405841/FRONT \
  --scene-graph swin-2 \
  --prefilter none \
  --symmetrize

python build_pairs.py --root data_p2_sparse
"""


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

IMAGE_EXTS_DEFAULT = [".png", ".jpg", ".jpeg"]
IMAGE_DIRNAMES_DEFAULT = ["rgb", "images"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image pairs for sparse-view scenes.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory, e.g. /path/to/data_p2_sparse",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default="swin-2",
        help="Pair strategy using DUSt3R semantics: complete, swin[-k][-noncyclic], "
             "logwin[-k][-noncyclic], oneref[-i]",
    )
    parser.add_argument(
        "--prefilter",
        type=str,
        default="none",
        help="Optional filter: none, seqN, cycN (e.g. seq2, cyc3)",
    )
    parser.add_argument(
        "--symmetrize",
        dest="symmetrize",
        action="store_true",
        help="Match DUSt3R's symmetrized inference mode by also saving reversed pairs.",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help="Disable symmetric reversed pairs.",
    )
    parser.set_defaults(symmetrize=True)
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
        "--rgb-dirname",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="pairs",
        help="Name of output folder under each scene directory.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=IMAGE_EXTS_DEFAULT,
        help="Allowed image extensions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated.",
    )
    return parser.parse_args()


def natural_key(path: Path):
    """
    Sort filenames like 000000.png, 000010.png, 000198.png naturally.
    """
    s = path.stem
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return key


def resolve_image_dirnames(args: argparse.Namespace) -> List[str]:
    if args.rgb_dirname:
        return [args.rgb_dirname]
    return list(dict.fromkeys(args.image_dirnames))


def discover_image_dirs(root: Path, image_dirnames: Sequence[str], exts: Sequence[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    image_dirnames = set(image_dirnames)
    image_dirs = []
    for p in root.rglob("*"):
        if not p.is_dir() or p.name not in image_dirnames:
            continue
        has_images = any(f.is_file() and f.suffix.lower() in exts for f in p.iterdir())
        if has_images:
            image_dirs.append(p)
    image_dirs.sort()
    return image_dirs


def list_images(image_dir: Path, exts: Sequence[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    imgs = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=natural_key)
    return imgs


def parse_scene_graph(scene_graph: str) -> Tuple[str, int | None, bool]:
    sg = scene_graph.strip().lower()
    if sg == "complete":
        return "complete", None, False

    if sg.startswith("swin"):
        is_cyclic = not sg.endswith("noncyclic")
        try:
            winsize = int(sg.split("-")[1])
        except Exception:
            winsize = 3
        return "swin", winsize, is_cyclic

    if sg.startswith("logwin"):
        is_cyclic = not sg.endswith("noncyclic")
        try:
            winsize = int(sg.split("-")[1])
        except Exception:
            winsize = 3
        return "logwin", winsize, is_cyclic

    if sg.startswith("oneref"):
        ref = int(sg.split("-")[1]) if "-" in sg else 0
        return "oneref", ref, False

    raise ValueError(
        f"Unsupported --scene-graph={scene_graph}. "
        f"Use complete, swin[-k][-noncyclic], logwin[-k][-noncyclic], or oneref[-i]."
    )


def _filter_edges_seq(edges: Sequence[Tuple[int, int]], seq_dis_thr: int, cyclic: bool = False) -> List[int]:
    n = max(max(e) for e in edges) + 1
    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def keep_edge_by_prefilter(i: int, j: int, num_images: int, prefilter: str) -> bool:
    pf = prefilter.strip().lower()
    if pf in ("none", "", "null"):
        return True

    m_seq = re.fullmatch(r"seq(\d+)", pf)
    if m_seq:
        n = int(m_seq.group(1))
        return abs(i - j) <= n

    m_cyc = re.fullmatch(r"cyc(\d+)", pf)
    if m_cyc:
        n = int(m_cyc.group(1))
        cyc_dist = min(abs(i - j), num_images - abs(i - j))
        return cyc_dist <= n

    raise ValueError(
        f"Unsupported --prefilter={prefilter}. Use none, seqN, or cycN."
    )


def apply_prefilter(
    edges: Sequence[Tuple[int, int]],
    prefilter: str,
) -> List[Tuple[int, int]]:
    pf = prefilter.strip().lower()
    if pf in ("none", "", "null"):
        return list(edges)

    if pf.startswith("seq"):
        kept = _filter_edges_seq(edges, int(pf[3:]), cyclic=False)
        return [edges[i] for i in kept]

    if pf.startswith("cyc"):
        kept = _filter_edges_seq(edges, int(pf[3:]), cyclic=True)
        return [edges[i] for i in kept]

    raise ValueError(
        f"Unsupported --prefilter={prefilter}. Use none, seqN, or cycN."
    )


def maybe_symmetrize(
    edges: Iterable[Tuple[int, int]],
    symmetrize: bool,
) -> List[Tuple[int, int]]:
    if not symmetrize:
        return sorted(set(edges))

    out = set()
    for i, j in edges:
        out.add((i, j))
        out.add((j, i))
    return sorted(out)


def build_pair_indices(
    num_images: int,
    scene_graph: str,
    prefilter: str,
    symmetrize: bool,
) -> List[Tuple[int, int]]:
    if num_images < 2:
        return []

    mode, val, is_cyclic = parse_scene_graph(scene_graph)
    pairs: List[Tuple[int, int]] = []

    if mode == "complete":
        for i in range(num_images):
            for j in range(i):
                pairs.append((i, j))

    elif mode == "swin":
        pairsid = set()
        for i in range(num_images):
            for j in range(1, val + 1):
                idx = i + j
                if is_cyclic:
                    idx = idx % num_images
                if idx >= num_images:
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))
        pairs.extend(sorted(pairsid))

    elif mode == "logwin":
        offsets = [2 ** i for i in range(val)]
        pairsid = set()
        for i in range(num_images):
            for j in [i - off for off in offsets] + [i + off for off in offsets]:
                if is_cyclic:
                    j = j % num_images
                if j < 0 or j >= num_images or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))
        pairs.extend(sorted(pairsid))

    elif mode == "oneref":
        if not (0 <= val < num_images):
            raise ValueError(
                f"Reference index {val} out of range for {num_images} images."
            )
        for j in range(num_images):
            if j != val:
                pairs.append((val, j))

    else:
        raise ValueError(f"Unexpected scene_graph mode: {mode}")

    if symmetrize:
        pairs += [(j, i) for i, j in pairs]

    return apply_prefilter(pairs, prefilter)


def build_pairs(
    images: Sequence[Path],
    scene_graph: str,
    prefilter: str,
    symmetrize: bool,
) -> List[Tuple[int, int]]:
    return build_pair_indices(len(images), scene_graph, prefilter, symmetrize)


def frame_id_from_name(path: Path) -> int | str:
    """
    Try to parse integer frame id from filename stem.
    Falls back to the stem string if not numeric.
    """
    s = path.stem
    if s.isdigit():
        return int(s)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return s


def make_pair_records(
    root: Path,
    image_dir: Path,
    images: Sequence[Path],
    pairs: Sequence[Tuple[int, int]],
    scene_graph: str,
    prefilter: str,
    symmetrize: bool,
):
    scene_dir = image_dir.parent
    scene_rel = scene_dir.relative_to(root)

    records = []
    for pair_id, (i, j) in enumerate(pairs):
        img_i = images[i]
        img_j = images[j]
        frame_i = frame_id_from_name(img_i)
        frame_j = frame_id_from_name(img_j)

        if isinstance(frame_i, int) and isinstance(frame_j, int):
            dt = frame_j - frame_i
        else:
            dt = None

        records.append(
            {
                "pair_id": pair_id,
                "i": i,
                "j": j,
                "img_i": str(img_i.resolve()),
                "img_j": str(img_j.resolve()),
                "img_i_rel_to_scene": str(img_i.relative_to(scene_dir)),
                "img_j_rel_to_scene": str(img_j.relative_to(scene_dir)),
                "img_i_name": img_i.name,
                "img_j_name": img_j.name,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "delta_frame": dt,
                "scene_rel": str(scene_rel),
            }
        )

    return records, {
        "scene_rel": str(scene_rel),
        "scene_dir": str(scene_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "image_dir_name": image_dir.name,
        "num_images": len(images),
        "num_pairs": len(records),
        "scene_graph": scene_graph,
        "prefilter": prefilter,
        "symmetrize": symmetrize,
        "images": [str(p.name) for p in images],
    }


def save_pairs_for_scene(
    root: Path,
    image_dir: Path,
    images: Sequence[Path],
    pairs: Sequence[Tuple[int, int]],
    output_dirname: str,
    scene_graph: str,
    prefilter: str,
    symmetrize: bool,
    dry_run: bool,
) -> Tuple[List[dict], dict]:
    scene_dir = image_dir.parent
    scene_rel = scene_dir.relative_to(root)

    out_dir = scene_dir / output_dirname
    pairs_json_path = out_dir / "pairs.json"
    pairs_txt_path = out_dir / "pairs.txt"
    meta_path = out_dir / "pairs_meta.json"
    records, meta = make_pair_records(
        root=root,
        image_dir=image_dir,
        images=images,
        pairs=pairs,
        scene_graph=scene_graph,
        prefilter=prefilter,
        symmetrize=symmetrize,
    )

    print(f"[scene] {scene_rel}")
    print(f"  images   : {len(images)}")
    print(f"  pairs    : {len(records)}")
    print(f"  strategy : {scene_graph}, prefilter={prefilter}, symmetrize={symmetrize}")

    if dry_run:
        return records, meta

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(pairs_txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['img_i']} {r['img_j']}\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return records, meta


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    image_dirnames = resolve_image_dirnames(args)

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    image_dirs = discover_image_dirs(root, image_dirnames, args.exts)
    if not image_dirs:
        raise RuntimeError(
            f"No image folders named {image_dirnames} with supported files found under: {root}"
        )

    processed = 0
    skipped = 0

    for image_dir in image_dirs:
        scene_dir = image_dir.parent
        scene_rel = str(scene_dir.relative_to(root))

        if args.only_scene and args.only_scene not in scene_rel:
            skipped += 1
            continue

        images = list_images(image_dir, args.exts)
        if len(images) < 2:
            print(f"[skip] {scene_rel}: fewer than 2 images")
            skipped += 1
            continue

        pairs = build_pairs(
            images=images,
            scene_graph=args.scene_graph,
            prefilter=args.prefilter,
            symmetrize=args.symmetrize,
        )

        if not pairs:
            print(f"[skip] {scene_rel}: no pairs generated")
            skipped += 1
            continue

        save_pairs_for_scene(
            root=root,
            image_dir=image_dir,
            images=images,
            pairs=pairs,
            output_dirname=args.output_dirname,
            scene_graph=args.scene_graph,
            prefilter=args.prefilter,
            symmetrize=args.symmetrize,
            dry_run=args.dry_run,
        )
        processed += 1

    print("\nDone.")
    print(f"Processed scenes: {processed}")
    print(f"Skipped scenes  : {skipped}")


if __name__ == "__main__":
    main()
