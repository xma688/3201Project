#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.pipeline import build_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Part 3 hybrid scene and optionally launch training")
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument("--trajectory-manifest", type=str, required=True, help="Path to trajectory_manifest.json")
    parser.add_argument("--pseudo-manifest", type=str, required=True, help="Path to pseudo_manifest.json")
    parser.add_argument("--confidence-manifest", type=str, required=True, help="Path to confidence_manifest.json")
    parser.add_argument("--hybrid-name", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--iterations", type=int, default=0)
    args = parser.parse_args()
    result = build_hybrid(
        config_path=args.config or None,
        trajectory_manifest_path=args.trajectory_manifest,
        pseudo_manifest_path=args.pseudo_manifest,
        confidence_manifest_path=args.confidence_manifest,
        hybrid_name=args.hybrid_name or None,
        train=args.train,
        output_tag=args.output_tag or None,
        iterations=args.iterations or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
