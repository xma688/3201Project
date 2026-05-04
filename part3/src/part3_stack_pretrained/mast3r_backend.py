from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils_pretrained import confidence_from_patch_values


@dataclass(frozen=True)
class Mast3RMatchResult:
    confidence: np.ndarray
    num_matches: int
    match_density: float


class Mast3RBackend:
    """Offline MASt3R reciprocal-matching backend for C_feat/S_feat."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        image_size: int = 512,
        patch_size: int = 16,
        min_confidence: float = 0.05,
        subsample: int = 8,
        block_size: int = 8192,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device_name = str(device)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.min_confidence = float(min_confidence)
        self.subsample = int(subsample)
        self.block_size = int(block_size)
        self._torch: Any | None = None
        self._model: Any | None = None
        self._inference: Any | None = None
        self._load_images: Any | None = None
        self._fast_nn: Any | None = None

    @property
    def available(self) -> bool:
        return self.repo_root.is_dir() and self.checkpoint_path.is_file()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                "MASt3R backend is not available. Expected repo/checkpoint at "
                f"{self.repo_root}, {self.checkpoint_path}."
            )
        dust3r_root = self.repo_root / "dust3r"
        if not (dust3r_root / "dust3r").is_dir():
            project_root_candidate = self.repo_root.parents[1] / "dust3r"
            if (project_root_candidate / "dust3r").is_dir():
                dust3r_root = project_root_candidate
                # MASt3R imports this helper for side effects. If the submodule
                # checkout is empty, we provide the equivalent side effect here.
                sys.modules.setdefault("mast3r.utils.path_to_dust3r", types.ModuleType("mast3r.utils.path_to_dust3r"))
        for item in (str(self.repo_root), str(dust3r_root)):
            if item not in sys.path:
                sys.path.insert(0, item)
        try:
            import torch  # type: ignore
            import mast3r.utils.path_to_dust3r  # noqa: F401
            from dust3r.inference import inference  # type: ignore
            from dust3r.utils.image import load_images  # type: ignore
            from mast3r.fast_nn import fast_reciprocal_NNs  # type: ignore
            from mast3r.model import AsymmetricMASt3R  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Failed to import MASt3R dependencies. Run this entry with an environment "
                "that has MASt3R and DUSt3R requirements installed."
            ) from exc

        device = torch.device(self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu")
        model = AsymmetricMASt3R.from_pretrained(str(self.checkpoint_path)).to(device).eval()
        self._torch = torch
        self._model = model
        self._device = device
        self._inference = inference
        self._load_images = load_images
        self._fast_nn = fast_reciprocal_NNs

    def compute_confidence(
        self,
        *,
        pseudo_path: str | Path,
        reference_path: str | Path,
        target_shape: tuple[int, int],
        valid_mask: np.ndarray | None = None,
    ) -> Mast3RMatchResult:
        self._ensure_model()
        torch = self._torch
        model = self._model
        inference = self._inference
        load_images = self._load_images
        fast_reciprocal_NNs = self._fast_nn
        assert torch is not None and model is not None and inference is not None and load_images is not None and fast_reciprocal_NNs is not None

        with torch.no_grad():
            images = load_images([str(Path(pseudo_path).expanduser()), str(Path(reference_path).expanduser())], size=self.image_size)
            output = inference([tuple(images)], model, self._device, batch_size=1, verbose=False)
            view1, pred1 = output["view1"], output["pred1"]
            view2, pred2 = output["view2"], output["pred2"]
            desc1 = pred1["desc"].squeeze(0).detach()
            desc2 = pred2["desc"].squeeze(0).detach()
            matches0, matches1 = fast_reciprocal_NNs(
                desc1,
                desc2,
                subsample_or_initxy1=self.subsample,
                device=self._device,
                dist="dot",
                block_size=self.block_size,
            )

        matches0_np = _to_numpy(matches0)
        matches1_np = _to_numpy(matches1)
        h0, w0 = _true_shape(view1)
        h1, w1 = _true_shape(view2)
        valid = (
            (matches0_np[:, 0] >= 3)
            & (matches0_np[:, 0] < max(0, w0 - 3))
            & (matches0_np[:, 1] >= 3)
            & (matches0_np[:, 1] < max(0, h0 - 3))
            & (matches1_np[:, 0] >= 3)
            & (matches1_np[:, 0] < max(0, w1 - 3))
            & (matches1_np[:, 1] >= 3)
            & (matches1_np[:, 1] < max(0, h1 - 3))
        )
        matches0_np = matches0_np[valid]
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        if h0 > 0 and w0 > 0 and matches0_np.size:
            scaled = matches0_np.copy().astype(np.float32)
            scaled[:, 0] *= target_w / float(w0)
            scaled[:, 1] *= target_h / float(h0)
        else:
            scaled = np.zeros((0, 2), dtype=np.float32)

        confidence = confidence_from_patch_values(
            (target_h, target_w),
            scaled,
            patch_size=self.patch_size,
            min_value=self.min_confidence,
            valid_mask=valid_mask,
        )
        density = float(len(scaled) / max(1, (target_h * target_w) / max(1, self.patch_size * self.patch_size)))
        return Mast3RMatchResult(confidence=confidence, num_matches=int(len(scaled)), match_density=density)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _true_shape(view: dict[str, Any]) -> tuple[int, int]:
    shape = view.get("true_shape")
    if shape is None:
        img = view.get("img")
        if img is not None and hasattr(img, "shape"):
            return int(img.shape[-2]), int(img.shape[-1])
        return 0, 0
    if hasattr(shape, "detach"):
        arr = shape.detach().cpu().numpy()
    else:
        arr = np.asarray(shape)
    arr = arr.reshape(-1)
    if arr.size >= 2:
        return int(arr[0]), int(arr[1])
    return 0, 0
