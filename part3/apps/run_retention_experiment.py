#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.retention import (  # noqa: E402
    DEFAULT_RETENTION_THRESHOLDS,
    DEFAULT_SCAN_NUM_INTERMEDIATE_VIEWS,
    format_markdown_summary,
    run_retention_experiment,
)


def _default_config_path() -> Path:
    part3_root = Path(__file__).resolve().parents[1]
    full_config = part3_root / "configs" / "project_gen_full.json"
    return full_config if full_config.exists() else part3_root / "configs" / "project.json"


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer, e.g. 4,6,8")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Part 3 pseudo-view retention scan up to confidence/consistency only."
    )
    parser.add_argument("--config", type=str, default=str(_default_config_path()), help="Defaults to project_gen_full.json")
    parser.add_argument("--scene", type=str, default="Re10k-1")
    parser.add_argument("--num-intermediate-views", type=_parse_int_list, default=list(DEFAULT_SCAN_NUM_INTERMEDIATE_VIEWS))
    parser.add_argument("--max-pairs", type=int, default=0, help="Use 0 for all adjacent sparse-frame pairs")
    parser.add_argument("--scan-keep-ratio", type=float, default=1.0)
    parser.add_argument("--validate-keep-ratio", type=float, default=0.8)
    parser.add_argument("--skip-keep-ratio-validation", action="store_true")
    parser.add_argument("--run-prefix", type=str, default="", help="Defaults to '<scene>_full_maxpairs<max_pairs>'")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--dust3r-variant", type=str, default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=-1.0)
    parser.add_argument("--fs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clip-score-threshold", type=float, default=DEFAULT_RETENTION_THRESHOLDS["clip_score"])
    parser.add_argument(
        "--mean-raw-confidence-threshold",
        type=float,
        default=DEFAULT_RETENTION_THRESHOLDS["mean_raw_confidence"],
    )
    parser.add_argument(
        "--patch-keep-ratio-threshold",
        type=float,
        default=DEFAULT_RETENTION_THRESHOLDS["patch_keep_ratio"],
    )
    parser.add_argument(
        "--reprojection-valid-ratio-threshold",
        type=float,
        default=DEFAULT_RETENTION_THRESHOLDS["reprojection_valid_ratio"],
    )
    parser.add_argument("--stable-ratio-threshold", type=float, default=0.65)
    parser.add_argument("--n8-min-extra-retained-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--summary-only", action="store_true", help="Only summarize existing manifests; do not run stages")
    parser.add_argument("--no-reuse-existing", action="store_true", help="Re-run stages even if manifests already exist")
    parser.add_argument("--disable-clip-consistency", action="store_true")
    parser.add_argument("--disable-patch-pruning", action="store_true")
    args = parser.parse_args()

    dynami_crafter = {"seed": args.seed}
    if args.steps > 0:
        dynami_crafter["steps"] = args.steps
    if args.cfg_scale > 0:
        dynami_crafter["cfg_scale"] = args.cfg_scale
    if args.eta >= 0:
        dynami_crafter["eta"] = args.eta
    if args.fs > 0:
        dynami_crafter["fs"] = args.fs

    thresholds = {
        "clip_score": args.clip_score_threshold,
        "mean_raw_confidence": args.mean_raw_confidence_threshold,
        "patch_keep_ratio": args.patch_keep_ratio_threshold,
        "reprojection_valid_ratio": args.reprojection_valid_ratio_threshold,
    }
    payload = run_retention_experiment(
        config_path=args.config,
        scene=args.scene,
        num_intermediate_views=args.num_intermediate_views,
        max_pairs=args.max_pairs,
        scan_keep_ratio=args.scan_keep_ratio,
        validate_keep_ratio=None if args.skip_keep_ratio_validation else args.validate_keep_ratio,
        run_prefix=args.run_prefix or None,
        prompt=args.prompt or None,
        dynami_crafter=dynami_crafter,
        dust3r_variant=args.dust3r_variant or None,
        thresholds=thresholds,
        reuse_existing=not args.no_reuse_existing,
        summary_only=args.summary_only,
        enable_clip_consistency=not args.disable_clip_consistency,
        enable_patch_pruning=not args.disable_patch_pruning,
        stable_ratio_threshold=args.stable_ratio_threshold,
        n8_min_extra_retained_ratio=args.n8_min_extra_retained_ratio,
        output_dir=args.output_dir or None,
    )
    print(format_markdown_summary(payload))
    print(json.dumps({"outputs": payload["outputs"], "decision": payload["decision"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
