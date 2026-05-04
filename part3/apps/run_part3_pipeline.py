#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.config import load_config
from part3_stack.pipeline import run_part3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Part 3 pipeline directly without HTTP services")
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument("--scene", type=str, required=True, help="Scene key, e.g. Re10k-1")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for DynamiCrafter interpolation")
    parser.add_argument("--num-intermediate-views", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--dust3r-variant", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--hybrid-name", type=str, default="")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=-1.0)
    parser.add_argument("--fs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--keep-ratio", type=float, default=-1.0)
    parser.add_argument("--border-margin-ratio", type=float, default=-1.0)
    parser.add_argument("--flow-sigma", type=float, default=-1.0)
    parser.add_argument("--anchor-sigma", type=float, default=-1.0)
    parser.add_argument("--reproj-sigma", type=float, default=-1.0)
    parser.add_argument("--alpha-threshold", type=float, default=-1.0)
    parser.add_argument("--boundary-percentile", type=float, default=-1.0)
    parser.add_argument("--boundary-dilate", type=int, default=-1)
    parser.add_argument("--feature-backend", type=str, default="")
    parser.add_argument("--feature-sigma", type=float, default=-1.0)
    parser.add_argument("--enable-clip-consistency", action="store_true", default=None)
    parser.add_argument("--disable-clip-consistency", action="store_false", dest="enable_clip_consistency")
    parser.add_argument("--enable-patch-pruning", action="store_true", default=None)
    parser.add_argument("--disable-patch-pruning", action="store_false", dest="enable_patch_pruning")
    parser.add_argument("--patch-size", type=int, default=-1)
    parser.add_argument("--patch-threshold", type=float, default=-1.0)
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

    response = run_part3_pipeline(
        config_path=config.config_path,
        scene=args.scene,
        prompt=args.prompt or config.defaults.get("prompt"),
        num_intermediate_views=args.num_intermediate_views,
        max_pairs=args.max_pairs,
        dust3r_variant=args.dust3r_variant or None,
        run_id=args.run_id or None,
        dynami_crafter=dynami_crafter or None,
        border_margin_ratio=args.border_margin_ratio if args.border_margin_ratio >= 0 else None,
        flow_sigma=args.flow_sigma if args.flow_sigma >= 0 else None,
        anchor_sigma=args.anchor_sigma if args.anchor_sigma >= 0 else None,
        hybrid_name=args.hybrid_name or None,
        output_tag=args.output_tag or None,
        train=args.train,
        evaluate=args.evaluate,
        iterations=args.iterations or None,
        keep_ratio=args.keep_ratio if args.keep_ratio >= 0 else None,
        reproj_sigma=args.reproj_sigma if args.reproj_sigma >= 0 else None,
        alpha_threshold=args.alpha_threshold if args.alpha_threshold >= 0 else None,
        boundary_percentile=args.boundary_percentile if args.boundary_percentile >= 0 else None,
        boundary_dilate=args.boundary_dilate if args.boundary_dilate >= 0 else None,
        feature_backend=args.feature_backend or None,
        feature_sigma=args.feature_sigma if args.feature_sigma >= 0 else None,
        enable_clip_consistency=args.enable_clip_consistency,
        enable_patch_pruning=args.enable_patch_pruning,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        patch_threshold=args.patch_threshold if args.patch_threshold >= 0 else None,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
