#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack_pretrained.pipeline_pretrained import build_confidence_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Part 3 pretrained MASt3R + SEA-RAFT confidence maps")
    parser.add_argument("--config", type=str, default="", help="Path to pretrained project config")
    parser.add_argument("--pseudo-manifest", type=str, required=True, help="Path to pseudo_manifest.json")
    parser.add_argument("--border-margin-ratio", type=float, default=-1.0)
    parser.add_argument("--reproj-sigma", type=float, default=-1.0)
    parser.add_argument("--alpha-threshold", type=float, default=-1.0)
    parser.add_argument("--boundary-percentile", type=float, default=-1.0)
    parser.add_argument("--boundary-dilate", type=int, default=-1)
    parser.add_argument("--enable-clip-consistency", action="store_true", default=None)
    parser.add_argument("--disable-clip-consistency", action="store_false", dest="enable_clip_consistency")
    parser.add_argument("--enable-patch-pruning", action="store_true", default=None)
    parser.add_argument("--disable-patch-pruning", action="store_false", dest="enable_patch_pruning")
    parser.add_argument("--patch-size", type=int, default=-1)
    parser.add_argument("--patch-threshold", type=float, default=-1.0)
    args = parser.parse_args()
    result = build_confidence_pretrained(
        config_path=args.config or None,
        pseudo_manifest_path=args.pseudo_manifest,
        border_margin_ratio=args.border_margin_ratio if args.border_margin_ratio >= 0 else None,
        reproj_sigma=args.reproj_sigma if args.reproj_sigma >= 0 else None,
        alpha_threshold=args.alpha_threshold if args.alpha_threshold >= 0 else None,
        boundary_percentile=args.boundary_percentile if args.boundary_percentile >= 0 else None,
        boundary_dilate=args.boundary_dilate if args.boundary_dilate >= 0 else None,
        enable_clip_consistency=args.enable_clip_consistency,
        enable_patch_pruning=args.enable_patch_pruning,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        patch_threshold=args.patch_threshold if args.patch_threshold >= 0 else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
