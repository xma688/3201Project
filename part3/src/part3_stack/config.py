from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneSpec:
    key: str
    scene_rel: Path
    image_dir: str
    sparse_step: int
    aliases: tuple[str, ...]


SCENES: tuple[SceneSpec, ...] = (
    SceneSpec(
        key="405841_FRONT",
        scene_rel=Path("405841/FRONT"),
        image_dir="images",
        sparse_step=10,
        aliases=("405841_FRONT", "405841/FRONT", "405841", "waymo_front"),
    ),
    SceneSpec(
        key="DL3DV-2",
        scene_rel=Path("DL3DV-2"),
        image_dir="images",
        sparse_step=30,
        aliases=("DL3DV-2", "dl3dv"),
    ),
    SceneSpec(
        key="Re10k-1",
        scene_rel=Path("Re10k-1"),
        image_dir="images",
        sparse_step=30,
        aliases=("Re10k-1", "re10k"),
    ),
)


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    project_root: Path
    part3_root: Path
    gaussian_splatting_root: Path
    dust3r_to_colmap_root: Path
    dust3r_outputs_root: Path
    data_p2_sparse_root: Path
    dynami_crafter_root: Path
    dynami_crafter_interp_ckpt: Path
    workspace_root: Path
    defaults: dict[str, Any]


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "project.json"


def _to_path(raw: str | Path, base: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    cfg_path = _to_path(config_path or default_config_path())
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    project_root = _to_path(payload["project_root"], base=cfg_path.parent)

    def project_path(key: str) -> Path:
        return _to_path(payload[key], base=project_root)

    return ProjectConfig(
        config_path=cfg_path,
        project_root=project_root,
        part3_root=project_path("part3_root"),
        gaussian_splatting_root=project_path("gaussian_splatting_root"),
        dust3r_to_colmap_root=project_path("dust3r_to_colmap_root"),
        dust3r_outputs_root=project_path("dust3r_outputs_root"),
        data_p2_sparse_root=project_path("data_p2_sparse_root"),
        dynami_crafter_root=project_path("dynami_crafter_root"),
        dynami_crafter_interp_ckpt=project_path("dynami_crafter_interp_ckpt"),
        workspace_root=project_path("workspace_root"),
        defaults=dict(payload["defaults"]),
    )


def normalize_scene_key(scene_name: str) -> str:
    lowered = scene_name.strip().lower()
    for scene in SCENES:
        if lowered == scene.key.lower() or lowered in {alias.lower() for alias in scene.aliases}:
            return scene.key
    raise KeyError(f"Unsupported scene: {scene_name}")


def get_scene_spec(scene_name: str) -> SceneSpec:
    key = normalize_scene_key(scene_name)
    for scene in SCENES:
        if scene.key == key:
            return scene
    raise KeyError(scene_name)


def ensure_workspace_dirs(config: ProjectConfig) -> None:
    for rel in (
        "runs",
        "pseudo_views",
        "confidence_maps",
        "hybrid_scenes",
        "3dgs_outputs",
    ):
        (config.workspace_root / rel).mkdir(parents=True, exist_ok=True)


def make_run_id(scene_key: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{scene_key}_{ts}"


def scene_run_dir(config: ProjectConfig, scene_key: str, run_id: str) -> Path:
    return config.workspace_root / "runs" / scene_key / run_id


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
