#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.pipeline import prepare_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Part 3 base scene and interpolated trajectory manifest")
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument("--scene", type=str, required=True, help="Scene key, e.g. Re10k-1")
    parser.add_argument("--num-intermediate-views", type=int, default=0, help="Override default number of interpolated views")
    parser.add_argument("--max-pairs", type=int, default=None, help="Override default number of anchor pairs; use 0 for all adjacent sparse-frame pairs")
    parser.add_argument("--all-adjacent-pairs", action="store_true", help="Interpolate every adjacent pair in the sparse trajectory")
    parser.add_argument("--dust3r-variant", type=str, default="", help="Preferred DUSt3R-to-COLMAP variant")
    parser.add_argument("--run-id", type=str, default="", help="Optional custom run id")
    args = parser.parse_args()
    result = prepare_scene(
        config_path=args.config or None,
        scene=args.scene,
        num_intermediate_views=args.num_intermediate_views or None,
        max_pairs=0 if args.all_adjacent_pairs else args.max_pairs,
        dust3r_variant=args.dust3r_variant or None,
        run_id=args.run_id or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
