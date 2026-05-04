from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ProjectConfig

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _validate_dynamicrafter_resolution(repo_root: Path, resolution: str) -> tuple[int, int]:
    try:
        height_text, width_text = resolution.split("_", 1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DynamiCrafter resolution '{resolution}'. Expected format like '384_512'."
        ) from exc

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid DynamiCrafter resolution '{resolution}': H/W must be positive.")

    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            f"Invalid DynamiCrafter resolution '{resolution}': H/W must be multiples of 64. "
            "Otherwise the UNet down/up sampling path can hit skip-connection shape mismatches. "
            "Use a stable size such as '320_512', '384_512', or '576_1024'."
        )

    config_file = repo_root / "configs" / f"inference_{width}_v1.0.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"DynamiCrafter config for width {width} was not found: {config_file}. "
            "Use a width with an available inference config, e.g. 512 or 1024."
        )
    return height, width


class DynamiCrafterInterpolationAdapter:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._model_resolution: str | None = None
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def _working_directory(self, path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    def _ensure_checkpoint(self) -> None:
        ckpt_dir = self.config.dynami_crafter_root / "checkpoints" / "dynamicrafter_512_interp_v1"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / "model.ckpt"
        source = self.config.dynami_crafter_interp_ckpt
        if target.exists():
            return
        if not source.exists():
            raise FileNotFoundError(f"DynamiCrafter interpolation checkpoint not found: {source}")
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)

    def _get_model(self, resolution: str) -> Any:
        _validate_dynamicrafter_resolution(self.config.dynami_crafter_root, resolution)
        if self._model is not None and self._model_resolution == resolution:
            return self._model

        self._ensure_checkpoint()
        repo_root = self.config.dynami_crafter_root
        repo_str = str(repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        with self._working_directory(repo_root):
            module = importlib.import_module("scripts.gradio.i2v_test_application")
            self._model = module.Image2Video(
                result_dir=str((self.config.workspace_root / "pseudo_views" / "_bootstrap").expanduser()),
                resolution=resolution,
            )
            self._model_resolution = resolution
        return self._model

    def generate_clip(
        self,
        start_image_path: str | Path,
        end_image_path: str | Path,
        prompt: str,
        output_dir: str | Path,
        steps: int,
        cfg_scale: float,
        eta: float,
        fs: int,
        seed: int,
        resolution: str = "384_512",
    ) -> Path:
        start_path = Path(start_image_path).expanduser()
        end_path = Path(end_image_path).expanduser()
        result_dir = Path(output_dir).expanduser()
        result_dir.mkdir(parents=True, exist_ok=True)

        image_a = np.array(Image.open(start_path).convert("RGB"))
        image_b = np.array(Image.open(end_path).convert("RGB"))

        with self._lock:
            model = self._get_model(str(resolution))
            model.result_dir = str(result_dir)
            with self._working_directory(self.config.dynami_crafter_root):
                video_path = model.get_image(
                    image=image_a,
                    prompt=prompt,
                    steps=int(steps),
                    cfg_scale=float(cfg_scale),
                    eta=float(eta),
                    fs=int(fs),
                    seed=int(seed),
                    image2=image_b,
                )
        return Path(video_path).expanduser()


def extract_uniform_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int,
    prefix: str,
) -> list[Path]:
    if num_frames <= 0:
        return []

    video = Path(video_path).expanduser()
    frame_dir = Path(output_dir).expanduser()
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames = _read_video_frames(video)
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video}")

    if len(frames) <= num_frames:
        indices = list(range(len(frames)))
    else:
        start_idx = 1 if len(frames) > 2 else 0
        end_idx = len(frames) - 2 if len(frames) > 2 else len(frames) - 1
        picks = np.linspace(start_idx, end_idx, num_frames)
        indices = [int(round(v)) for v in picks]

    paths: list[Path] = []
    for out_idx, frame_idx in enumerate(indices[:num_frames]):
        out_path = frame_dir / f"{prefix}_{out_idx:02d}.png"
        Image.fromarray(frames[frame_idx]).save(out_path)
        paths.append(out_path)
    return paths


def _read_video_frames(video: Path) -> list[np.ndarray]:
    errors: list[str] = []

    if imageio is not None:
        try:
            reader = imageio.get_reader(video)
            try:
                return [np.asarray(frame) for frame in reader]
            finally:
                reader.close()
        except Exception as exc:
            errors.append(f"imageio: {exc}")
    else:
        errors.append("imageio: module not installed")

    try:
        return _read_video_frames_with_torchvision(video)
    except Exception as exc:
        errors.append(f"torchvision: {exc}")

    try:
        return _read_video_frames_with_cv2(video)
    except Exception as exc:
        errors.append(f"cv2: {exc}")

    try:
        return _read_video_frames_with_ffmpeg(video)
    except Exception as exc:
        errors.append(f"ffmpeg: {exc}")

    raise RuntimeError(
        "Failed to decode generated video. Install one video reader in the Part 3 environment "
        "(recommended: `pip install imageio imageio-ffmpeg`, or install OpenCV/ffmpeg). "
        f"Video path: {video}. Tried: {'; '.join(errors)}"
    )


def _read_video_frames_with_torchvision(video: Path) -> list[np.ndarray]:
    from torchvision.io import read_video  # type: ignore

    frames, _audio, _info = read_video(str(video), pts_unit="sec")
    return [frame.numpy() for frame in frames]


def _read_video_frames_with_cv2(video: Path) -> list[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed.") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open generated video: {video}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _read_video_frames_with_ffmpeg(video: Path) -> list[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable is not available on PATH.")

    with tempfile.TemporaryDirectory(prefix="part3_decode_") as tmp:
        frame_pattern = str(Path(tmp) / "frame_%06d.png")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            frame_pattern,
        ]
        subprocess.run(command, check=True)
        return [np.asarray(Image.open(path).convert("RGB")) for path in sorted(Path(tmp).glob("frame_*.png"))]
