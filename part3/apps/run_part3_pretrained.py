#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.config import load_config
from part3_stack_pretrained.pipeline_pretrained import run_part3_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Part 3 pretrained MASt3R + SEA-RAFT route")
    parser.add_argument("--config", type=str, default="", help="Path to pretrained project config")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--pseudo-manifest", type=str, default="", help="Reuse an existing pseudo_manifest.json")
    parser.add_argument("--trajectory-manifest", type=str, default="", help="Optional trajectory manifest override")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--num-intermediate-views", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--hybrid-name", type=str, default="")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=-1.0)
    parser.add_argument("--fs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config or None)
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

    result = run_part3_pretrained(
        config_path=config.config_path,
        scene=args.scene,
        pseudo_manifest_path=args.pseudo_manifest or None,
        trajectory_manifest_path=args.trajectory_manifest or None,
        prompt=args.prompt or config.defaults.get("prompt"),
        num_intermediate_views=args.num_intermediate_views,
        max_pairs=args.max_pairs,
        run_id=args.run_id or None,
        dynami_crafter=dynami_crafter or None,
        keep_ratio=args.keep_ratio if args.keep_ratio >= 0 else None,
        hybrid_name=args.hybrid_name or None,
        train=args.train,
        output_tag=args.output_tag or None,
        iterations=args.iterations or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
