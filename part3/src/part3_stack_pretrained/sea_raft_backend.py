from __future__ import annotations

import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeaRaftResult:
    flow: np.ndarray
    info: np.ndarray | None


class SeaRaftBackend:
    """Thin offline wrapper around the official SEA-RAFT inference code."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        cfg_path: str | Path,
        device: str = "cuda",
        iters: int | None = None,
        use_uncertainty: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.cfg_path = Path(cfg_path).expanduser()
        self.device_name = str(device)
        self.iters_override = iters
        self.use_uncertainty = bool(use_uncertainty)
        self._torch: Any | None = None
        self._model: Any | None = None
        self._args: Namespace | None = None

    @property
    def available(self) -> bool:
        return self.repo_root.is_dir() and self.checkpoint_path.is_file() and self.cfg_path.is_file()

    def _load_args(self) -> Namespace:
        data = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        args = Namespace(**data)
        args.path = str(self.checkpoint_path)
        args.url = None
        args.device = self.device_name
        if self.iters_override is not None:
            args.iters = int(self.iters_override)
        return args

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                "SEA-RAFT backend is not available. Expected repo/checkpoint/config at "
                f"{self.repo_root}, {self.checkpoint_path}, {self.cfg_path}."
            )
        core_root = self.repo_root / "core"
        for item in (str(core_root), str(self.repo_root)):
            if item not in sys.path:
                sys.path.insert(0, item)
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            import extractor  # type: ignore
            from raft import RAFT  # type: ignore
            from utils.utils import load_ckpt  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Failed to import SEA-RAFT dependencies. Run this entry with an environment "
                "that has torch and SEA-RAFT requirements installed."
            ) from exc

        args = self._load_args()
        device = torch.device(self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu")
        original_init_weights = extractor.ResNetFPN._init_weights
        extractor.ResNetFPN._init_weights = _offline_resnet_fpn_init_weights(nn)
        try:
            model = RAFT(args)
        finally:
            extractor.ResNetFPN._init_weights = original_init_weights
        load_ckpt(model, str(self.checkpoint_path))
        model = model.to(device).eval()
        self._torch = torch
        self._model = model
        self._args = args
        self._device = device

    def _to_tensor(self, rgb: np.ndarray) -> Any:
        torch = self._torch
        assert torch is not None
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {arr.shape}")
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(self._device)

    def predict(self, first: np.ndarray, second: np.ndarray) -> SeaRaftResult:
        self._ensure_model()
        torch = self._torch
        model = self._model
        args = self._args
        assert torch is not None and model is not None and args is not None
        image1 = self._to_tensor(first)
        image2 = self._to_tensor(second)
        with torch.no_grad():
            if int(getattr(args, "scale", 0)) != 0:
                scale = float(2 ** int(args.scale))
                image1_work = torch.nn.functional.interpolate(image1, scale_factor=scale, mode="bilinear", align_corners=False)
                image2_work = torch.nn.functional.interpolate(image2, scale_factor=scale, mode="bilinear", align_corners=False)
                output = model(image1_work, image2_work, iters=int(args.iters), test_mode=True)
                flow = torch.nn.functional.interpolate(output["flow"][-1], size=image1.shape[-2:], mode="bilinear", align_corners=False) / scale
                info = torch.nn.functional.interpolate(output["info"][-1], size=image1.shape[-2:], mode="area")
            else:
                output = model(image1, image2, iters=int(args.iters), test_mode=True)
                flow = output["flow"][-1]
                info = output["info"][-1]
        return SeaRaftResult(
            flow=flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32),
            info=info[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32),
        )

    def pair_confidence(
        self,
        first: np.ndarray,
        second: np.ndarray,
        *,
        sigma: float,
        uncertainty_sigma: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        result_ab = self.predict(first, second)
        result_ba = self.predict(second, first)
        fb_error = _forward_backward_error(result_ab.flow, result_ba.flow)
        conf = np.exp(-fb_error / max(1e-3, float(sigma))).astype(np.float32)
        if self.use_uncertainty and result_ab.info is not None:
            heat = _sea_raft_heatmap(result_ab.info)
            if np.isfinite(heat).any() and float(np.max(heat)) > 1e-6:
                heat = heat / max(1e-6, float(np.percentile(heat, 95.0)))
                conf *= np.exp(-heat / max(1e-3, float(uncertainty_sigma))).astype(np.float32)
        conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
        return conf, {
            "mean_fb_error": float(np.mean(fb_error)),
            "mean_confidence": float(np.mean(conf)),
        }


def _forward_backward_error(flow_ab: np.ndarray, flow_ba: np.ndarray) -> np.ndarray:
    h, w = flow_ab.shape[:2]
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise ImportError("OpenCV is required to warp SEA-RAFT backward flow for consistency.") from exc
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_ab[..., 0]
    map_y = grid_y + flow_ab[..., 1]
    warped_ba = cv2.remap(flow_ba, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return np.linalg.norm(flow_ab + warped_ba, axis=2).astype(np.float32)


def _offline_resnet_fpn_init_weights(nn_module: Any) -> Any:
    """Mirror SEA-RAFT's local init while skipping torchvision ImageNet downloads."""

    def init_weights(resnet_fpn: Any, _args: Any) -> None:
        for module in resnet_fpn.modules():
            if isinstance(module, nn_module.Conv2d):
                nn_module.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn_module.init.constant_(module.bias, 0)
            elif isinstance(module, (nn_module.BatchNorm2d, nn_module.InstanceNorm2d, nn_module.GroupNorm)):
                if module.weight is not None:
                    nn_module.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn_module.init.constant_(module.bias, 0)

    return init_weights


def _sea_raft_heatmap(info: np.ndarray) -> np.ndarray:
    if info.shape[-1] < 4:
        return np.zeros(info.shape[:2], dtype=np.float32)
    raw_b = info[..., 2:4]
    weights = _softmax(info[..., :2], axis=-1)
    log_b0 = np.clip(raw_b[..., 0], 0.0, 10.0)
    log_b1 = np.clip(raw_b[..., 1], 0.0, 10.0)
    return np.maximum(weights[..., 0] * log_b0 + weights[..., 1] * log_b1, 0.0).astype(np.float32)


def _softmax(array: np.ndarray, axis: int) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-8)
