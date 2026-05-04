from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any

from .config import ProjectConfig, ensure_workspace_dirs, load_config, read_json, scene_run_dir, write_json
from .pipeline import build_confidence, generate_pseudo_views, prepare_scene


DEFAULT_RETENTION_THRESHOLDS = {
    "clip_score": 0.55,
    "mean_raw_confidence": 0.10,
    "patch_keep_ratio": 0.20,
    "reprojection_valid_ratio": 0.70,
}

DEFAULT_SCAN_NUM_INTERMEDIATE_VIEWS = (4, 6, 8)


def _ratio_suffix(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "kr" + text.replace(".", "p")


def make_retention_run_id(
    *,
    scene: str,
    num_intermediate_views: int,
    keep_ratio: float,
    max_pairs: int,
    run_prefix: str | None = None,
) -> str:
    prefix = run_prefix or f"{scene}_full_maxpairs{int(max_pairs)}"
    return f"{prefix}_N{int(num_intermediate_views)}_{_ratio_suffix(keep_ratio)}"


def _split_kept_views(
    view_records: list[dict[str, Any]],
    keep_ratio: float,
    min_keep_per_clip: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not view_records:
        return [], []
    clamped_ratio = min(1.0, max(0.0, float(keep_ratio)))
    keep_count = len(view_records) if clamped_ratio >= 0.999 else max(
        int(min_keep_per_clip),
        int(round(len(view_records) * clamped_ratio)),
    )
    keep_count = max(1, min(len(view_records), keep_count))
    selected_names = {
        item["output_name"]
        for item in sorted(view_records, key=lambda record: float(record.get("frame_score", 1.0)))[:keep_count]
    }
    kept = [{**record, "filtered_out": False} for record in view_records if record["output_name"] in selected_names]
    dropped = [{**record, "filtered_out": True} for record in view_records if record["output_name"] not in selected_names]
    return kept, dropped


def derive_pseudo_manifest_with_keep_ratio(
    *,
    config: ProjectConfig,
    source_pseudo_manifest_path: str | Path,
    output_run_id: str,
    keep_ratio: float,
    min_keep_per_clip: int | None = None,
) -> Path:
    """Build a new pseudo manifest by re-filtering an existing keep_ratio=1 run.

    This avoids regenerating DynamiCrafter clips when validating whether the
    pre-confidence pseudo filter should use keep_ratio=0.8.
    """

    source_manifest = read_json(source_pseudo_manifest_path)
    scene = str(source_manifest["scene"])
    output_run_dir = scene_run_dir(config, scene, output_run_id)
    default_min_keep = int(config.defaults.get("pseudo_filter", {}).get("min_keep_per_clip", 1))
    min_keep = default_min_keep if min_keep_per_clip is None else int(min_keep_per_clip)
    source_trajectory_path = Path(source_manifest["trajectory_manifest_path"]).expanduser()
    trajectory_manifest_path = source_trajectory_path
    if source_trajectory_path.exists():
        trajectory_payload = {
            **read_json(source_trajectory_path),
            "run_id": output_run_id,
            "run_dir": str(output_run_dir),
        }
        trajectory_manifest_path = write_json(output_run_dir / "trajectory_manifest.json", trajectory_payload)

    clips: list[dict[str, Any]] = []
    for clip in source_manifest["clips"]:
        all_views = [
            {key: value for key, value in item.items() if key != "filtered_out"}
            for item in list(clip.get("assigned_views", [])) + list(clip.get("filtered_views", []))
        ]
        kept_views, filtered_views = _split_kept_views(all_views, keep_ratio, min_keep)
        clips.append(
            {
                **clip,
                "assigned_views": kept_views,
                "filtered_views": filtered_views,
                "num_generated_frames": int(clip.get("num_generated_frames", len(all_views))),
                "num_kept_frames": len(kept_views),
                "derived_from_pseudo_manifest": str(Path(source_pseudo_manifest_path).expanduser()),
            }
        )

    payload = {
        **source_manifest,
        "run_id": output_run_id,
        "run_dir": str(output_run_dir),
        "trajectory_manifest_path": str(trajectory_manifest_path),
        "keep_ratio": float(keep_ratio),
        "derived_from_pseudo_manifest": str(Path(source_pseudo_manifest_path).expanduser()),
        "clips": clips,
    }
    return write_json(output_run_dir / "pseudo_views" / "pseudo_manifest.json", payload)


def _manifest_paths(config: ProjectConfig, scene: str, run_id: str) -> dict[str, Path]:
    run_dir = scene_run_dir(config, scene, run_id)
    return {
        "run_dir": run_dir,
        "trajectory_manifest_path": run_dir / "trajectory_manifest.json",
        "pseudo_manifest_path": run_dir / "pseudo_views" / "pseudo_manifest.json",
        "confidence_manifest_path": run_dir / "confidence" / "confidence_manifest.json",
    }


def _passes_retention(record: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        float(record.get("clip_score", 0.0)) >= float(thresholds["clip_score"])
        and float(record.get("mean_raw_confidence", 0.0)) >= float(thresholds["mean_raw_confidence"])
        and float(record.get("patch_keep_ratio", 0.0)) >= float(thresholds["patch_keep_ratio"])
        and float(record.get("reprojection_valid_ratio", 0.0)) >= float(thresholds["reprojection_valid_ratio"])
    )


def _safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def summarize_retention_run(
    *,
    trajectory_manifest_path: str | Path,
    pseudo_manifest_path: str | Path,
    confidence_manifest_path: str | Path,
    thresholds: dict[str, float] | None = None,
    trial_type: str = "scan",
) -> dict[str, Any]:
    thresholds = {**DEFAULT_RETENTION_THRESHOLDS, **(thresholds or {})}
    trajectory = read_json(trajectory_manifest_path)
    pseudo_manifest = read_json(pseudo_manifest_path)
    confidence_manifest = read_json(confidence_manifest_path)

    pseudo_by_pair = {clip["pair_id"]: clip for clip in pseudo_manifest.get("clips", [])}
    total_generated = sum(int(clip.get("num_generated_frames", 0)) for clip in pseudo_by_pair.values())
    if total_generated <= 0:
        total_generated = len(trajectory.get("pairs", [])) * int(trajectory.get("num_intermediate_views", 0))
    pseudo_filter_kept = sum(int(clip.get("num_kept_frames", len(clip.get("assigned_views", [])))) for clip in pseudo_by_pair.values())

    all_records: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []
    for clip in confidence_manifest.get("clips", []):
        pair_id = clip["pair_id"]
        pseudo_clip = pseudo_by_pair.get(pair_id, {})
        generated_for_pair = int(pseudo_clip.get("num_generated_frames", len(clip.get("mask_records", []))))
        retained_for_pair = 0
        clip_records: list[dict[str, Any]] = []
        for mask_record in clip.get("mask_records", []):
            record = {
                **mask_record,
                "pair_id": pair_id,
                "retained_by_thresholds": _passes_retention(mask_record, thresholds),
            }
            retained_for_pair += int(record["retained_by_thresholds"])
            clip_records.append(record)
            all_records.append(record)

        pair_ratio = retained_for_pair / generated_for_pair if generated_for_pair else 0.0
        pair_stats.append(
            {
                "pair_id": pair_id,
                "generated": generated_for_pair,
                "pseudo_filter_kept": int(pseudo_clip.get("num_kept_frames", len(pseudo_clip.get("assigned_views", [])))),
                "post_confidence_kept": retained_for_pair,
                "post_confidence_keep_ratio": pair_ratio,
                "mean_confidence": (
                    sum(float(item.get("mean_confidence", 0.0)) for item in clip_records) / len(clip_records)
                    if clip_records
                    else 0.0
                ),
                "clip_score": float(clip.get("clip_metrics", {}).get("clip_score", 0.0)),
            }
        )

    post_confidence_kept = sum(int(record["retained_by_thresholds"]) for record in all_records)
    post_confidence_keep_ratio = post_confidence_kept / total_generated if total_generated else 0.0
    post_confidence_keep_ratio_vs_pseudo_filter = (
        post_confidence_kept / pseudo_filter_kept if pseudo_filter_kept else 0.0
    )
    worst_pair = min(
        pair_stats,
        key=lambda item: (
            float(item["post_confidence_keep_ratio"]),
            float(item["mean_confidence"]),
            float(item["clip_score"]),
        ),
        default={"pair_id": ""},
    )

    return {
        "trial_type": trial_type,
        "scene": trajectory.get("scene", pseudo_manifest.get("scene", "")),
        "run_id": pseudo_manifest.get("run_id", trajectory.get("run_id", "")),
        "num_intermediate_views": int(trajectory.get("num_intermediate_views", 0)),
        "max_pairs": int(trajectory.get("max_pairs", 0)),
        "num_anchor_pairs": len(trajectory.get("pairs", [])),
        "keep_ratio": float(pseudo_manifest.get("keep_ratio", 0.0)),
        "generated": int(total_generated),
        "pseudo_filter_kept": int(pseudo_filter_kept),
        "post_confidence_kept": int(post_confidence_kept),
        "post_confidence_keep_ratio": float(post_confidence_keep_ratio),
        "post_confidence_keep_ratio_vs_pseudo_filter": float(post_confidence_keep_ratio_vs_pseudo_filter),
        "median_mean_confidence": _safe_median([float(record.get("mean_confidence", 0.0)) for record in all_records]),
        "median_mean_raw_confidence": _safe_median([float(record.get("mean_raw_confidence", 0.0)) for record in all_records]),
        "median_patch_keep_ratio": _safe_median([float(record.get("patch_keep_ratio", 0.0)) for record in all_records]),
        "median_clip_score": _safe_median([float(record.get("clip_score", 0.0)) for record in all_records]),
        "worst_pair_id": str(worst_pair.get("pair_id", "")),
        "thresholds": thresholds,
        "pair_stats": pair_stats,
        "paths": {
            "trajectory_manifest_path": str(Path(trajectory_manifest_path).expanduser()),
            "pseudo_manifest_path": str(Path(pseudo_manifest_path).expanduser()),
            "confidence_manifest_path": str(Path(confidence_manifest_path).expanduser()),
        },
    }


def recommend_num_intermediate_views(
    rows: list[dict[str, Any]],
    *,
    stable_ratio_threshold: float = 0.65,
    n8_min_extra_retained_ratio: float = 0.15,
) -> dict[str, Any]:
    scan_rows = {int(row["num_intermediate_views"]): row for row in rows if row.get("trial_type") == "scan"}
    row4 = scan_rows.get(4)
    row6 = scan_rows.get(6)
    row8 = scan_rows.get(8)
    rationale: list[str] = []

    selected: int | None = None
    if row6 and float(row6["post_confidence_keep_ratio"]) >= stable_ratio_threshold:
        selected = 6
        rationale.append(
            f"N=6 keep ratio is {row6['post_confidence_keep_ratio']:.3f}, meeting the {stable_ratio_threshold:.2f} threshold."
        )
        if row8 and int(row6["post_confidence_kept"]) > 0:
            extra = (int(row8["post_confidence_kept"]) - int(row6["post_confidence_kept"])) / int(
                row6["post_confidence_kept"]
            )
            if extra < n8_min_extra_retained_ratio:
                rationale.append(
                    f"N=8 adds only {extra:.1%} retained frames over N=6, so N=8 is not worth the extra generation cost."
                )
            else:
                rationale.append(
                    f"N=8 adds {extra:.1%} retained frames over N=6; use it only if the extra runtime is acceptable."
                )
    elif row6 and row4 and float(row4["post_confidence_keep_ratio"]) >= stable_ratio_threshold:
        selected = 4
        rationale.append(
            f"N=6 falls below {stable_ratio_threshold:.2f}, while N=4 remains stable at {row4['post_confidence_keep_ratio']:.3f}."
        )
    elif scan_rows:
        best_row = max(
            scan_rows.values(),
            key=lambda row: (float(row["post_confidence_keep_ratio"]), int(row["post_confidence_kept"])),
        )
        selected = int(best_row["num_intermediate_views"])
        rationale.append(
            f"No default rule matched; selected N={selected} because it has the best retained ratio/count among scanned runs."
        )
    else:
        rationale.append("No scan rows were available for a recommendation.")

    return {
        "recommended_num_intermediate_views": selected,
        "recommended_validation_keep_ratio": 0.8 if selected is not None else None,
        "stable_ratio_threshold": float(stable_ratio_threshold),
        "n8_min_extra_retained_ratio": float(n8_min_extra_retained_ratio),
        "rationale": rationale,
    }


def _run_or_reuse_trial(
    *,
    config: ProjectConfig,
    scene: str,
    num_intermediate_views: int,
    keep_ratio: float,
    max_pairs: int,
    run_id: str,
    prompt: str | None,
    dynami_crafter: dict[str, Any] | None,
    dust3r_variant: str | None,
    thresholds: dict[str, float],
    reuse_existing: bool,
    summary_only: bool,
    enable_clip_consistency: bool,
    enable_patch_pruning: bool,
    trial_type: str,
) -> dict[str, Any]:
    paths = _manifest_paths(config, scene, run_id)
    confidence_exists = paths["confidence_manifest_path"].exists()
    if summary_only:
        missing = [str(path) for key, path in paths.items() if key != "run_dir" and not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot summarize {run_id}; missing manifests: {missing}")
    elif not (reuse_existing and confidence_exists):
        if not (reuse_existing and paths["trajectory_manifest_path"].exists()):
            prepare_scene(
                config_path=config.config_path,
                scene=scene,
                num_intermediate_views=num_intermediate_views,
                max_pairs=max_pairs,
                dust3r_variant=dust3r_variant,
                run_id=run_id,
            )
        if not (reuse_existing and paths["pseudo_manifest_path"].exists()):
            generate_pseudo_views(
                config_path=config.config_path,
                trajectory_manifest_path=paths["trajectory_manifest_path"],
                prompt=prompt,
                dynami_crafter=dynami_crafter,
                keep_ratio=keep_ratio,
            )
        if not (reuse_existing and paths["confidence_manifest_path"].exists()):
            build_confidence(
                config_path=config.config_path,
                pseudo_manifest_path=paths["pseudo_manifest_path"],
                enable_clip_consistency=enable_clip_consistency,
                enable_patch_pruning=enable_patch_pruning,
            )

    return summarize_retention_run(
        trajectory_manifest_path=paths["trajectory_manifest_path"],
        pseudo_manifest_path=paths["pseudo_manifest_path"],
        confidence_manifest_path=paths["confidence_manifest_path"],
        thresholds=thresholds,
        trial_type=trial_type,
    )


def _run_or_reuse_derived_validation(
    *,
    config: ProjectConfig,
    scene: str,
    num_intermediate_views: int,
    source_run_id: str,
    validation_run_id: str,
    keep_ratio: float,
    thresholds: dict[str, float],
    reuse_existing: bool,
    summary_only: bool,
    enable_clip_consistency: bool,
    enable_patch_pruning: bool,
) -> dict[str, Any]:
    source_paths = _manifest_paths(config, scene, source_run_id)
    validation_paths = _manifest_paths(config, scene, validation_run_id)
    if summary_only:
        missing = [
            str(validation_paths[key])
            for key in ("pseudo_manifest_path", "confidence_manifest_path")
            if not validation_paths[key].exists()
        ]
        if missing:
            raise FileNotFoundError(f"Cannot summarize {validation_run_id}; missing manifests: {missing}")
    elif not (reuse_existing and validation_paths["confidence_manifest_path"].exists()):
        if not source_paths["pseudo_manifest_path"].exists():
            raise FileNotFoundError(f"Cannot derive keep_ratio={keep_ratio}; missing {source_paths['pseudo_manifest_path']}")
        if not (reuse_existing and validation_paths["pseudo_manifest_path"].exists()):
            derive_pseudo_manifest_with_keep_ratio(
                config=config,
                source_pseudo_manifest_path=source_paths["pseudo_manifest_path"],
                output_run_id=validation_run_id,
                keep_ratio=keep_ratio,
            )
        if not (reuse_existing and validation_paths["confidence_manifest_path"].exists()):
            build_confidence(
                config_path=config.config_path,
                pseudo_manifest_path=validation_paths["pseudo_manifest_path"],
                enable_clip_consistency=enable_clip_consistency,
                enable_patch_pruning=enable_patch_pruning,
            )

    return summarize_retention_run(
        trajectory_manifest_path=validation_paths["trajectory_manifest_path"]
        if validation_paths["trajectory_manifest_path"].exists()
        else source_paths["trajectory_manifest_path"],
        pseudo_manifest_path=validation_paths["pseudo_manifest_path"],
        confidence_manifest_path=validation_paths["confidence_manifest_path"],
        thresholds=thresholds,
        trial_type="keep_ratio_validation",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trial_type",
        "num_intermediate_views",
        "keep_ratio",
        "generated",
        "pseudo_filter_kept",
        "post_confidence_kept",
        "post_confidence_keep_ratio",
        "median_mean_confidence",
        "median_patch_keep_ratio",
        "median_clip_score",
        "worst_pair_id",
        "run_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_markdown_summary(payload: dict[str, Any]) -> str:
    headers = [
        "trial",
        "N",
        "keep_ratio",
        "generated",
        "pseudo_filter_kept",
        "post_conf_kept",
        "post_conf_ratio",
        "median_conf",
        "median_patch",
        "median_clip",
        "worst_pair",
    ]
    lines = [
        "# Part 3 Pseudo-View Retention Experiment",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in payload["rows"]:
        values = [
            str(row["trial_type"]),
            str(row["num_intermediate_views"]),
            f"{row['keep_ratio']:.3g}",
            str(row["generated"]),
            str(row["pseudo_filter_kept"]),
            str(row["post_confidence_kept"]),
            f"{row['post_confidence_keep_ratio']:.3f}",
            f"{row['median_mean_confidence']:.3f}",
            f"{row['median_patch_keep_ratio']:.3f}",
            f"{row['median_clip_score']:.3f}",
            str(row["worst_pair_id"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    decision = payload["decision"]
    lines.extend(["", "## Recommendation", ""])
    lines.append(f"- Recommended `num_intermediate_views`: `{decision['recommended_num_intermediate_views']}`")
    lines.append(f"- Recommended validation keep ratio: `{decision['recommended_validation_keep_ratio']}`")
    for reason in decision["rationale"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## Thresholds", ""])
    for key, value in payload["thresholds"].items():
        lines.append(f"- `{key}` >= `{value}`")
    return "\n".join(lines) + "\n"


def write_retention_outputs(
    *,
    config: ProjectConfig,
    scene: str,
    output_dir: str | Path | None,
    payload: dict[str, Any],
) -> dict[str, str]:
    target_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else config.workspace_root / "runs" / scene / "retention_experiment"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "retention_summary.csv"
    md_path = target_dir / "retention_summary.md"
    output_paths = {
        "json": str(target_dir / "retention_summary.json"),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    payload["outputs"] = output_paths
    json_path = write_json(output_paths["json"], payload)
    _write_csv(csv_path, payload["rows"])
    md_path.write_text(format_markdown_summary(payload), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def run_retention_experiment(
    *,
    config_path: str | Path,
    scene: str = "Re10k-1",
    num_intermediate_views: list[int] | tuple[int, ...] = DEFAULT_SCAN_NUM_INTERMEDIATE_VIEWS,
    max_pairs: int = 0,
    scan_keep_ratio: float = 1.0,
    validate_keep_ratio: float | None = 0.8,
    run_prefix: str | None = None,
    prompt: str | None = None,
    dynami_crafter: dict[str, Any] | None = None,
    dust3r_variant: str | None = None,
    thresholds: dict[str, float] | None = None,
    reuse_existing: bool = True,
    summary_only: bool = False,
    enable_clip_consistency: bool = True,
    enable_patch_pruning: bool = True,
    stable_ratio_threshold: float = 0.65,
    n8_min_extra_retained_ratio: float = 0.15,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_workspace_dirs(config)
    thresholds = {**DEFAULT_RETENTION_THRESHOLDS, **(thresholds or {})}
    rows: list[dict[str, Any]] = []
    scan_run_ids: dict[int, str] = {}

    for n_views in num_intermediate_views:
        run_id = make_retention_run_id(
            scene=scene,
            num_intermediate_views=int(n_views),
            keep_ratio=scan_keep_ratio,
            max_pairs=max_pairs,
            run_prefix=run_prefix,
        )
        scan_run_ids[int(n_views)] = run_id
        rows.append(
            _run_or_reuse_trial(
                config=config,
                scene=scene,
                num_intermediate_views=int(n_views),
                keep_ratio=scan_keep_ratio,
                max_pairs=max_pairs,
                run_id=run_id,
                prompt=prompt,
                dynami_crafter=dynami_crafter,
                dust3r_variant=dust3r_variant,
                thresholds=thresholds,
                reuse_existing=reuse_existing,
                summary_only=summary_only,
                enable_clip_consistency=enable_clip_consistency,
                enable_patch_pruning=enable_patch_pruning,
                trial_type="scan",
            )
        )

    decision = recommend_num_intermediate_views(
        rows,
        stable_ratio_threshold=stable_ratio_threshold,
        n8_min_extra_retained_ratio=n8_min_extra_retained_ratio,
    )
    if decision["recommended_num_intermediate_views"] is not None:
        decision["recommended_validation_keep_ratio"] = validate_keep_ratio
    selected_n = decision["recommended_num_intermediate_views"]
    if validate_keep_ratio is not None and selected_n is not None:
        validation_run_id = make_retention_run_id(
            scene=scene,
            num_intermediate_views=int(selected_n),
            keep_ratio=float(validate_keep_ratio),
            max_pairs=max_pairs,
            run_prefix=run_prefix,
        )
        rows.append(
            _run_or_reuse_derived_validation(
                config=config,
                scene=scene,
                num_intermediate_views=int(selected_n),
                source_run_id=scan_run_ids[int(selected_n)],
                validation_run_id=validation_run_id,
                keep_ratio=float(validate_keep_ratio),
                thresholds=thresholds,
                reuse_existing=reuse_existing,
                summary_only=summary_only,
                enable_clip_consistency=enable_clip_consistency,
                enable_patch_pruning=enable_patch_pruning,
            )
        )

    payload = {
        "scene": scene,
        "config_path": str(config.config_path),
        "max_pairs": int(max_pairs),
        "scan_keep_ratio": float(scan_keep_ratio),
        "validate_keep_ratio": validate_keep_ratio,
        "thresholds": thresholds,
        "decision": decision,
        "rows": rows,
    }
    write_retention_outputs(
        config=config,
        scene=scene,
        output_dir=output_dir,
        payload=payload,
    )
    return payload
