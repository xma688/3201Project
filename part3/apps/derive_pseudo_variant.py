#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part3_stack.config import load_config, read_json, scene_run_dir, write_json


def _absolute_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _build_derived_from(
    *,
    source_trajectory: dict[str, Any],
    source_pseudo: dict[str, Any],
    source_trajectory_path: Path,
    source_pseudo_path: Path,
) -> dict[str, Any]:
    return {
        "source_run_id": source_trajectory.get("run_id", source_pseudo.get("run_id")),
        "source_run_dir": source_trajectory.get("run_dir", source_pseudo.get("run_dir")),
        "source_trajectory_manifest_path": str(source_trajectory_path),
        "source_pseudo_manifest_path": str(source_pseudo_path),
    }


def derive_pseudo_variant(
    *,
    config_path: str | Path | None,
    source_trajectory_manifest_path: str | Path,
    source_pseudo_manifest_path: str | Path,
    target_run_id: str,
    target_run_dir: str | Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    source_trajectory_path = _absolute_path(source_trajectory_manifest_path)
    source_pseudo_path = _absolute_path(source_pseudo_manifest_path)
    _require_file(source_trajectory_path, "source trajectory manifest")
    _require_file(source_pseudo_path, "source pseudo manifest")

    source_trajectory = read_json(source_trajectory_path)
    source_pseudo = read_json(source_pseudo_path)
    scene = str(source_trajectory.get("scene") or source_pseudo.get("scene"))
    if not scene:
        raise ValueError("Could not infer scene from source manifests.")
    if str(source_pseudo.get("scene", scene)) != scene:
        raise ValueError(
            f"Scene mismatch: trajectory scene={scene}, pseudo scene={source_pseudo.get('scene')}"
        )

    resolved_target_run_dir = (
        _absolute_path(target_run_dir)
        if target_run_dir
        else scene_run_dir(config, scene, target_run_id)
    )
    target_trajectory_path = resolved_target_run_dir / "trajectory_manifest.json"
    target_pseudo_path = resolved_target_run_dir / "pseudo_views" / "pseudo_manifest.json"

    if not overwrite:
        existing = [path for path in (target_trajectory_path, target_pseudo_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing derived manifest(s): {joined}")

    derived_from = _build_derived_from(
        source_trajectory=source_trajectory,
        source_pseudo=source_pseudo,
        source_trajectory_path=source_trajectory_path,
        source_pseudo_path=source_pseudo_path,
    )

    target_trajectory = copy.deepcopy(source_trajectory)
    target_trajectory["run_id"] = target_run_id
    target_trajectory["run_dir"] = str(resolved_target_run_dir)
    target_trajectory["derived_from"] = derived_from

    target_pseudo = copy.deepcopy(source_pseudo)
    target_pseudo["run_id"] = target_run_id
    target_pseudo["run_dir"] = str(resolved_target_run_dir)
    target_pseudo["trajectory_manifest_path"] = str(target_trajectory_path)
    target_pseudo["derived_from"] = derived_from

    write_json(target_trajectory_path, target_trajectory)
    write_json(target_pseudo_path, target_pseudo)

    return {
        "scene": scene,
        "target_run_id": target_run_id,
        "target_run_dir": str(resolved_target_run_dir),
        "trajectory_manifest_path": str(target_trajectory_path),
        "pseudo_manifest_path": str(target_pseudo_path),
        "derived_from": derived_from,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a Part 3 pseudo-view variant from one shared generated pseudo run."
    )
    parser.add_argument("--config", type=str, default="", help="Path to project.json")
    parser.add_argument(
        "--source-trajectory-manifest",
        type=str,
        required=True,
        help="Source trajectory_manifest.json from the shared pseudo run",
    )
    parser.add_argument(
        "--source-pseudo-manifest",
        type=str,
        required=True,
        help="Source pseudo_views/pseudo_manifest.json from the shared pseudo run",
    )
    parser.add_argument("--target-run-id", type=str, required=True)
    parser.add_argument(
        "--target-run-dir",
        type=str,
        default="",
        help="Optional output run directory. Defaults to workspace/runs/<scene>/<target-run-id>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only the small derived manifest JSON files if they already exist.",
    )
    args = parser.parse_args()

    result = derive_pseudo_variant(
        config_path=args.config or None,
        source_trajectory_manifest_path=args.source_trajectory_manifest,
        source_pseudo_manifest_path=args.source_pseudo_manifest,
        target_run_id=args.target_run_id,
        target_run_dir=args.target_run_dir or None,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
