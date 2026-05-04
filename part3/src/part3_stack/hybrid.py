from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ProjectConfig, read_json, write_json
from .geometry import _load_colmap_io


def _copy_base_scene(base_scene_dir: Path, output_scene_dir: Path) -> None:
    if output_scene_dir.exists():
        shutil.rmtree(output_scene_dir)
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    for name in ("images", "sparse"):
        src = base_scene_dir / name
        if src.exists():
            shutil.copytree(src, output_scene_dir / name)

    for filename in ("transforms_train.json", "transforms_test.json", "transforms.json", "test.txt"):
        src = base_scene_dir / filename
        if src.exists():
            shutil.copy2(src, output_scene_dir / filename)


def _rgba_from_frame(frame_path: str | Path, mask_path: str | Path, width: int, height: int) -> Image.Image:
    rgb = Image.open(frame_path).convert("RGB").resize((width, height), Image.BILINEAR)
    alpha = np.load(mask_path)
    alpha_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.resize((width, height), Image.BILINEAR)
    rgba = Image.merge("RGBA", (*rgb.split(), alpha_img))
    return rgba


def _read_test_names(path: Path, valid_real_names: set[str]) -> list[str]:
    if not path.is_file():
        return []
    valid_by_stem = {Path(name).stem: name for name in valid_real_names if not name.startswith("pseudo_pair_")}
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw_name = line.strip()
        name = raw_name if raw_name in valid_real_names else valid_by_stem.get(Path(raw_name).stem, "")
        if name and not name.startswith("pseudo_pair_"):
            names.append(name)
    return names


def _real_test_names(output_scene_dir: Path, sparse_dir: Path, real_image_names: list[str], hold: int = 8) -> list[str]:
    valid_real_names = set(real_image_names)
    for candidate in (sparse_dir / "test.txt", output_scene_dir / "test.txt"):
        names = _read_test_names(candidate, valid_real_names)
        if names:
            return sorted(dict.fromkeys(names))
    hold = max(1, int(hold))
    return [name for idx, name in enumerate(sorted(real_image_names)) if idx % hold == 0]


def build_hybrid_scene(
    config: ProjectConfig,
    trajectory_manifest_path: str | Path,
    pseudo_manifest_path: str | Path,
    confidence_manifest_path: str | Path,
    hybrid_name: str,
) -> Path:
    trajectory = read_json(trajectory_manifest_path)
    pseudo_manifest = read_json(pseudo_manifest_path)
    confidence_manifest = read_json(confidence_manifest_path)

    base_scene_dir = Path(trajectory["base_scene_dir"]).expanduser()
    output_scene_dir = config.workspace_root / "hybrid_scenes" / hybrid_name
    _copy_base_scene(base_scene_dir, output_scene_dir)

    sparse_dir = output_scene_dir / "sparse" / "0"
    images_dir = output_scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    colmap_io = _load_colmap_io(config.project_root)
    cameras, images, points3d = colmap_io.read_model(str(sparse_dir), ext="")
    next_image_id = max(images.keys()) + 1 if images else 1
    real_image_names = sorted(str(image.name) for image in images.values() if not str(image.name).startswith("pseudo_pair_"))
    test_names = _real_test_names(output_scene_dir, sparse_dir, real_image_names)
    test_name_set = set(test_names)
    train_real_names = [name for name in real_image_names if name not in test_name_set]

    confidence_lookup = {
        (clip["pair_id"], record["output_name"]): record
        for clip in confidence_manifest["clips"]
        for record in clip["mask_records"]
    }

    appended_records: list[dict[str, Any]] = []
    for clip in pseudo_manifest["clips"]:
        for view_record in clip["assigned_views"]:
            confidence_record = confidence_lookup[(clip["pair_id"], view_record["output_name"])]
            width = int(view_record["width"])
            height = int(view_record["height"])
            out_name = view_record["output_name"]
            out_path = images_dir / out_name
            rgba = _rgba_from_frame(view_record["frame_path"], confidence_record["mask_path"], width, height)
            rgba.save(out_path)

            images[next_image_id] = colmap_io.Image(
                id=next_image_id,
                qvec=np.asarray(view_record["qvec"], dtype=np.float64),
                tvec=np.asarray(view_record["tvec"], dtype=np.float64),
                camera_id=int(view_record["camera_id"]),
                name=out_name,
                xys=np.zeros((0, 2), dtype=np.float64),
                point3D_ids=np.zeros((0,), dtype=np.int64),
            )
            appended_records.append(
                {
                    "image_id": next_image_id,
                    "image_name": out_name,
                    "image_path": str(out_path),
                    "camera_id": int(view_record["camera_id"]),
                    "qvec": view_record["qvec"],
                    "tvec": view_record["tvec"],
                    "mask_path": confidence_record["mask_path"],
                    "mean_confidence": confidence_record["mean_confidence"],
                    "mean_feature_confidence": confidence_record.get("mean_feature_confidence", 0.0),
                    "patch_keep_ratio": confidence_record.get("patch_keep_ratio", 1.0),
                    "clip_score": confidence_record.get("clip_score", 1.0),
                    "is_pseudo": True,
                    "split": "train",
                }
            )
            next_image_id += 1

    (sparse_dir / "test.txt").write_text("\n".join(test_names) + ("\n" if test_names else ""), encoding="utf-8")
    colmap_io.write_model(cameras, images, points3d, str(sparse_dir), ext=".txt")
    colmap_io.write_model(cameras, images, points3d, str(sparse_dir), ext=".bin")

    pseudo_views = [view for clip in pseudo_manifest["clips"] for view in clip["assigned_views"]]
    confidence_records = [record for clip in confidence_manifest["clips"] for record in clip["mask_records"]]
    total_generated = int(sum(int(clip.get("num_generated_frames", len(clip["assigned_views"]))) for clip in pseudo_manifest["clips"]))
    total_kept = int(len(pseudo_views))
    total_filtered = int(sum(len(clip.get("filtered_views", [])) for clip in pseudo_manifest["clips"]))
    mean_frame_score = float(np.mean([view.get("frame_score", 0.0) for view in pseudo_views])) if pseudo_views else 0.0
    mean_reprojection_error = float(np.mean([record.get("mean_reprojection_error", 0.0) for record in confidence_records])) if confidence_records else 0.0
    mean_confidence = float(np.mean([record.get("mean_confidence", 0.0) for record in confidence_records])) if confidence_records else 0.0
    mean_feature_confidence = float(np.mean([record.get("mean_feature_confidence", 0.0) for record in confidence_records])) if confidence_records else 0.0
    mean_patch_keep_ratio = float(np.mean([record.get("patch_keep_ratio", 1.0) for record in confidence_records])) if confidence_records else 0.0
    mean_clip_score = float(np.mean([record.get("clip_score", 1.0) for record in confidence_records])) if confidence_records else 0.0

    payload = {
        "scene": trajectory["scene"],
        "run_id": trajectory["run_id"],
        "base_scene_dir": str(base_scene_dir),
        "hybrid_scene_dir": str(output_scene_dir),
        "trajectory_manifest_path": str(Path(trajectory_manifest_path).expanduser()),
        "pseudo_manifest_path": str(Path(pseudo_manifest_path).expanduser()),
        "confidence_manifest_path": str(Path(confidence_manifest_path).expanduser()),
        "hybrid_name": hybrid_name,
        "appended_images": appended_records,
        "summary": {
            "num_generated_views": total_generated,
            "num_kept_views": total_kept,
            "num_filtered_views": total_filtered,
            "mean_frame_score": mean_frame_score,
            "mean_reprojection_error": mean_reprojection_error,
            "mean_confidence": mean_confidence,
            "mean_feature_confidence": mean_feature_confidence,
            "mean_patch_keep_ratio": mean_patch_keep_ratio,
            "mean_clip_score": mean_clip_score,
            "num_real_train_views": len(train_real_names),
            "num_real_test_views": len(test_names),
            "num_pseudo_train_views": total_kept,
            "num_pseudo_test_views": 0,
        },
        "splits": {
            "real_train": train_real_names,
            "real_test": test_names,
            "pseudo_train": [view["output_name"] for view in pseudo_views],
            "pseudo_test": [],
        },
    }
    return write_json(output_scene_dir / "part3_hybrid_manifest.json", payload)


def launch_training(
    config: ProjectConfig,
    scene_dir: str | Path,
    output_tag: str,
    iterations: int | None = None,
    confidence_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    script = config.part3_root / "scripts" / "train_part3_3dgs.sh"
    command = ["bash", str(script), str(Path(scene_dir).expanduser().resolve()), output_tag]
    if iterations is not None:
        command.append(str(iterations))
    if confidence_manifest_path is not None:
        command.extend(["--confidence_manifest", str(Path(confidence_manifest_path).expanduser().resolve())])
    training_defaults = dict(config.defaults.get("training", {}))
    online_defaults = dict(config.defaults.get("online_confidence", {}))
    diagnostics_defaults = dict(config.defaults.get("diagnostics", {}))
    consistency_defaults = dict(config.defaults.get("consistency", {}))
    if "pseudo_warmup_iters" in training_defaults:
        command.extend(["--pseudo_warmup_iters", str(int(training_defaults["pseudo_warmup_iters"]))])
    if "pseudo_full_iters" in training_defaults:
        command.extend(["--pseudo_full_iters", str(int(training_defaults["pseudo_full_iters"]))])
    if "pseudo_ratio_mid" in training_defaults:
        command.extend(["--pseudo_ratio_mid", str(float(training_defaults["pseudo_ratio_mid"]))])
    if "pseudo_ratio_final" in training_defaults:
        command.extend(["--pseudo_ratio_final", str(float(training_defaults["pseudo_ratio_final"]))])
    if "pseudo_weight_mid" in training_defaults:
        command.extend(["--pseudo_weight_mid", str(float(training_defaults["pseudo_weight_mid"]))])
    if "pseudo_weight_final" in training_defaults:
        command.extend(["--pseudo_weight_final", str(float(training_defaults["pseudo_weight_final"]))])
    if bool(training_defaults.get("enable_pseudo_lpips", False)):
        command.append("--enable_pseudo_lpips")
    if "pseudo_lpips_weight" in training_defaults:
        command.extend(["--pseudo_lpips_weight", str(float(training_defaults["pseudo_lpips_weight"]))])
    if bool(online_defaults.get("enabled", False)):
        command.append("--enable_online_confidence")
    if "refresh_interval" in online_defaults:
        command.extend(["--confidence_refresh_interval", str(int(online_defaults["refresh_interval"]))])
    if "writeback_interval" in online_defaults:
        command.extend(["--confidence_writeback_interval", str(int(online_defaults["writeback_interval"]))])
    if "rgb_sigma" in online_defaults:
        command.extend(["--online_rgb_sigma", str(float(online_defaults["rgb_sigma"]))])
    if "feature_sigma" in online_defaults:
        command.extend(["--online_feature_sigma", str(float(online_defaults["feature_sigma"]))])
    if "patch_size" in consistency_defaults:
        command.extend(["--online_patch_size", str(int(consistency_defaults["patch_size"]))])
    if "patch_threshold" in consistency_defaults:
        command.extend(["--online_patch_threshold", str(float(consistency_defaults["patch_threshold"]))])
    if "patch_low_weight" in consistency_defaults:
        command.extend(["--online_patch_low_weight", str(float(consistency_defaults["patch_low_weight"]))])
    if "patch_min_keep_ratio" in consistency_defaults:
        command.extend(["--online_patch_min_keep_ratio", str(float(consistency_defaults["patch_min_keep_ratio"]))])
    if "interval" in diagnostics_defaults:
        command.extend(["--diagnostics_interval", str(int(diagnostics_defaults["interval"]))])
    if "debug_views" in diagnostics_defaults:
        command.extend(["--diagnostics_debug_views", str(int(diagnostics_defaults["debug_views"]))])
    if not bool(diagnostics_defaults.get("export_png", True)):
        command.append("--disable_diagnostics_png")
    subprocess.run(command, check=True)
    return {
        "output_dir": str((config.workspace_root / "3dgs_outputs" / output_tag).expanduser()),
        "command": command,
    }


def launch_evaluation(config: ProjectConfig, model_dir: str | Path) -> dict[str, Any]:
    script = config.part3_root / "scripts" / "eval_part3_3dgs.sh"
    command = ["bash", str(script), str(Path(model_dir).expanduser())]
    subprocess.run(command, check=True)
    return {
        "model_dir": str(Path(model_dir).expanduser()),
        "command": command,
    }
