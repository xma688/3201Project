#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.pipeline import generate_pseudo_views


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Part 3 pseudo views from a trajectory manifest")
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument("--trajectory-manifest", type=str, required=True, help="Path to trajectory_manifest.json")
    parser.add_argument("--prompt", type=str, default="", help="Override generation prompt")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=-1.0)
    parser.add_argument("--fs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--resolution", type=str, default="", help="Override DynamiCrafter resolution, e.g. 384_512")
    parser.add_argument("--keep-ratio", type=float, default=-1.0, help="Keep the best ratio of pseudo frames after geometry-guided scoring")
    args = parser.parse_args()
    dynami_crafter = {}
    if args.steps > 0:
        dynami_crafter["steps"] = args.steps
    if args.cfg_scale > 0:
        dynami_crafter["cfg_scale"] = args.cfg_scale
    if args.eta >= 0:
        dynami_crafter["eta"] = args.eta
    if args.fs > 0:
        dynami_crafter["fs"] = args.fs
    if args.seed >= 0:
        dynami_crafter["seed"] = args.seed
    if args.resolution:
        dynami_crafter["resolution"] = args.resolution

    result = generate_pseudo_views(
        config_path=args.config or None,
        trajectory_manifest_path=args.trajectory_manifest,
        prompt=args.prompt or None,
        dynami_crafter=dynami_crafter or None,
        keep_ratio=args.keep_ratio if args.keep_ratio >= 0 else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
