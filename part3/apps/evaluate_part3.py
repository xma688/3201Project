#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.pipeline import evaluate_part3


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Part 3 3DGS model directory")
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument("--model-dir", type=str, default="", help="Explicit model directory to evaluate")
    parser.add_argument("--output-tag", type=str, default="", help="Model tag under workspace/3dgs_outputs")
    args = parser.parse_args()
    result = evaluate_part3(
        config_path=args.config or None,
        model_dir=args.model_dir or None,
        output_tag=args.output_tag or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
