#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GS_ROOT = PROJECT_ROOT / "gaussian-splatting"
if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from lpipsPyTorch.modules.lpips import LPIPS

    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


CONFIDENCE_FLOOR = 0.3
SOFT_CONFIDENCE_WEIGHTS = {
    "reproj": 0.4,
    "feat": 0.3,
    "temp": 0.3,
}


def apply_confidence_mask(image, gt_image, viewpoint_cam, enabled: bool):
    mask = get_confidence_mask(viewpoint_cam, enabled)
    if mask is None:
        return image, gt_image
    return image * mask, gt_image * mask


def get_confidence_mask(viewpoint_cam, enabled: bool):
    if not enabled:
        return None
    dynamic_mask = getattr(viewpoint_cam, "dynamic_confidence_mask", None)
    if dynamic_mask is not None:
        return dynamic_mask.cuda().clamp(0.0, 1.0)
    if viewpoint_cam.alpha_mask is None:
        return None
    return viewpoint_cam.alpha_mask.cuda().clamp(0.0, 1.0)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return torch.abs(pred - target).mean()
    weight = mask.expand_as(pred).clamp(0.0, 1.0)
    return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, normalize: bool, eps: float = 1e-8) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    weight = mask.expand_as(pred).clamp(0.0, 1.0)
    if normalize:
        return (diff2 * weight).sum() / weight.sum().clamp_min(eps)
    return (diff2 * weight).mean()


def psnr_from_mse(mse_value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 20.0 * torch.log10(torch.tensor(1.0, device=mse_value.device) / torch.sqrt(mse_value.clamp_min(eps)))


def is_pseudo_camera(viewpoint_cam) -> bool:
    return str(getattr(viewpoint_cam, "image_name", "")).startswith("pseudo_pair_")


def split_training_cameras(scene: Scene) -> tuple[list, list]:
    train_cameras = scene.getTrainCameras().copy()
    real_cameras = [cam for cam in train_cameras if not is_pseudo_camera(cam)]
    pseudo_cameras = [cam for cam in train_cameras if is_pseudo_camera(cam)]
    return real_cameras, pseudo_cameras


def select_diagnostic_cameras(real_cameras: list, pseudo_cameras: list, limit: int) -> list:
    limit = max(0, int(limit))
    if limit == 0:
        return []

    real_target = min(len(real_cameras), max(1, limit // 2)) if real_cameras else 0
    pseudo_target = min(len(pseudo_cameras), limit - real_target)
    if real_target + pseudo_target < limit:
        real_target = min(len(real_cameras), limit - pseudo_target)

    selected = list(real_cameras[:real_target]) + list(pseudo_cameras[:pseudo_target])
    if len(selected) < limit:
        for camera in list(real_cameras) + list(pseudo_cameras):
            if camera not in selected:
                selected.append(camera)
            if len(selected) >= limit:
                break
    return selected[:limit]


def pop_random_camera(pool: list, source: list):
    if not source:
        raise RuntimeError("Attempted to sample from an empty camera source list.")
    if not pool:
        pool.extend(source)
    idx = randint(0, len(pool) - 1)
    return pool.pop(idx)


def camera_name(viewpoint_cam) -> str:
    return str(getattr(viewpoint_cam, "image_name", ""))


def load_confidence_records(confidence_manifest_path: str | None) -> dict[str, dict]:
    if not confidence_manifest_path:
        return {}
    path = Path(confidence_manifest_path).expanduser()
    if not path.is_file():
        print(f"Confidence manifest not found: {path}. Continuing with PNG alpha masks only.")
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_ablation = dict(manifest.get("ablation", {}))
    records: dict[str, dict] = {}
    for clip in manifest.get("clips", []):
        clip_metrics = dict(clip.get("clip_metrics", {}))
        for record in clip.get("mask_records", []):
            merged = {**record, "clip_metrics": clip_metrics, "ablation": manifest_ablation}
            output_name = str(record.get("output_name", ""))
            if output_name:
                records[output_name] = merged
                records[Path(output_name).stem] = merged
    return records


def pop_weighted_pseudo_camera(pseudo_cameras: list, confidence_records: dict[str, dict]):
    if not pseudo_cameras:
        raise RuntimeError("Attempted to sample from an empty pseudo camera list.")
    weights = []
    for cam in pseudo_cameras:
        record = confidence_records.get(camera_name(cam), {})
        weights.append(max(0.05, float(record.get("clip_score", 1.0))))
    total = sum(weights)
    if total <= 0:
        return random.choice(pseudo_cameras)
    return random.choices(pseudo_cameras, weights=weights, k=1)[0]


def load_numpy_mask(path: str | None, target_hw: tuple[int, int], device: torch.device, clamp: bool = True) -> torch.Tensor | None:
    if not path:
        return None
    mask_path = Path(path).expanduser()
    if not mask_path.is_file():
        return None
    arr = np.load(mask_path).astype(np.float32)
    tensor = torch.from_numpy(arr)[None, None].to(device)
    if tuple(tensor.shape[-2:]) != target_hw:
        tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    tensor = tensor[0]
    return tensor.clamp(0.0, 1.0) if clamp else tensor


def normalize_depth_for_drift(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.float()
    finite = torch.isfinite(depth)
    if finite.any():
        valid = depth[finite]
        lo = torch.quantile(valid, 0.05)
        hi = torch.quantile(valid, 0.95)
        depth = (depth - lo) / torch.clamp(hi - lo, min=1e-6)
    return torch.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def gradient_feature_confidence(pred: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    pred_gray = pred.mean(dim=0, keepdim=True)
    target_gray = target.mean(dim=0, keepdim=True)
    pred_dx = F.pad(pred_gray[..., :, 1:] - pred_gray[..., :, :-1], (0, 1, 0, 0))
    pred_dy = F.pad(pred_gray[..., 1:, :] - pred_gray[..., :-1, :], (0, 0, 0, 1))
    target_dx = F.pad(target_gray[..., :, 1:] - target_gray[..., :, :-1], (0, 1, 0, 0))
    target_dy = F.pad(target_gray[..., 1:, :] - target_gray[..., :-1, :], (0, 0, 0, 1))
    pred_feat = torch.cat([pred, pred_gray, pred_dx, pred_dy], dim=0)
    target_feat = torch.cat([target, target_gray, target_dx, target_dy], dim=0)
    pred_feat = F.normalize(pred_feat, dim=0, eps=1e-6)
    target_feat = F.normalize(target_feat, dim=0, eps=1e-6)
    sim = ((pred_feat * target_feat).sum(dim=0, keepdim=True) + 1.0) * 0.5
    return torch.exp(-(1.0 - sim.clamp(0.0, 1.0)) / max(1e-3, float(sigma))).clamp(0.0, 1.0)


def torch_patch_pruning_mask(
    confidence: torch.Tensor,
    patch_size: int,
    threshold: float,
    low_weight: float,
    min_keep_ratio: float,
) -> tuple[torch.Tensor, float]:
    patch = max(1, int(patch_size))
    h, w = confidence.shape[-2:]
    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch
    padded = F.pad(confidence[None], (0, pad_w, 0, pad_h), mode="replicate")
    scores = F.avg_pool2d(padded, kernel_size=patch, stride=patch)
    keep = scores >= float(threshold)
    if keep.numel() and keep.float().mean().item() < float(min_keep_ratio):
        keep_count = max(1, int(np.ceil(keep.numel() * float(min_keep_ratio))))
        top_idx = torch.topk(scores.reshape(-1), k=keep_count).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.reshape(-1)[top_idx] = True
    up = F.interpolate(keep.float(), size=(padded.shape[-2], padded.shape[-1]), mode="nearest")[0, :, :h, :w]
    mask = float(low_weight) + (1.0 - float(low_weight)) * up
    return mask.clamp(0.0, 1.0), float((up > 0.5).float().mean().item())


def save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim == 3:
        image = image.permute(1, 2, 0)
    elif image.ndim == 2:
        image = image[..., None]
    arr = (image.numpy() * 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def maybe_refresh_online_confidence(
    *,
    viewpoint_cam,
    render_image: torch.Tensor,
    gt_image: torch.Tensor,
    render_depth: torch.Tensor | None,
    record: dict | None,
    iteration: int,
    enabled: bool,
    refresh_interval: int,
    writeback_interval: int,
    output_dir: Path,
    rgb_sigma: float,
    feature_sigma: float,
    patch_size: int,
    patch_threshold: float,
    patch_low_weight: float,
    patch_min_keep_ratio: float,
) -> dict[str, float]:
    if not enabled or record is None or refresh_interval <= 0 or iteration % refresh_interval != 0:
        return {}
    device = render_image.device
    h, w = render_image.shape[-2:]
    before = get_confidence_mask(viewpoint_cam, True)
    if before is None:
        before = torch.ones((1, h, w), device=device)
    else:
        before = before.to(device)

    flags = dict(record.get("ablation", {}))

    def component_enabled(name: str) -> bool:
        return bool(flags.get(name, True))

    hard_validity = torch.ones((1, h, w), device=device)
    c_vis = torch.ones((1, h, w), device=device)
    if component_enabled("use_c_vis"):
        loaded_hard = load_numpy_mask(record.get("hard_validity_path"), (h, w), device)
        if loaded_hard is not None:
            hard_validity = loaded_hard
        loaded = load_numpy_mask(record.get("c_vis_path"), (h, w), device)
        if loaded is not None:
            c_vis = loaded
    c_temp = torch.ones((1, h, w), device=device)
    if component_enabled("use_c_temp"):
        loaded = load_numpy_mask(record.get("c_temp_path"), (h, w), device)
        if loaded is not None:
            c_temp = loaded

    rgb_error = torch.abs(render_image.detach() - gt_image.detach()).mean(dim=0, keepdim=True)
    if component_enabled("use_c_reproj"):
        c_reproj = torch.exp(-rgb_error / max(1e-3, float(rgb_sigma))).clamp(0.0, 1.0)
    else:
        c_reproj = torch.ones((1, h, w), device=device)
    if component_enabled("use_c_feat"):
        c_feat = gradient_feature_confidence(render_image.detach(), gt_image.detach(), feature_sigma)
    else:
        c_feat = torch.ones((1, h, w), device=device)
    soft_confidence = (
        SOFT_CONFIDENCE_WEIGHTS["reproj"] * c_reproj
        + SOFT_CONFIDENCE_WEIGHTS["feat"] * c_feat
        + SOFT_CONFIDENCE_WEIGHTS["temp"] * c_temp
    ).clamp(0.0, 1.0)
    pre_patch_confidence = (hard_validity * (CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * soft_confidence)).clamp(0.0, 1.0)
    if component_enabled("use_patch_pruning"):
        c_patch, patch_keep_ratio = torch_patch_pruning_mask(
            pre_patch_confidence,
            patch_size=patch_size,
            threshold=patch_threshold,
            low_weight=patch_low_weight,
            min_keep_ratio=patch_min_keep_ratio,
        )
    else:
        c_patch = torch.ones((1, h, w), device=device)
        patch_keep_ratio = 1.0
    refreshed = (
        hard_validity * c_patch * (CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * soft_confidence)
    ).detach().clamp(0.0, 1.0)
    viewpoint_cam.dynamic_confidence_mask = refreshed

    depth_drift = 0.0
    coarse_depth = load_numpy_mask(record.get("coarse_depth_path"), (h, w), device, clamp=False)
    if coarse_depth is not None and render_depth is not None:
        pred_depth = normalize_depth_for_drift(render_depth.detach().float())
        coarse_norm = normalize_depth_for_drift(coarse_depth)
        depth_drift = torch.abs(pred_depth - coarse_norm).mean().item()

    metrics = {
        "mean_confidence_before": float(before.mean().item()),
        "mean_confidence_after": float(refreshed.mean().item()),
        "mean_reprojection_error_before": float(record.get("mean_reprojection_error", 0.0)),
        "mean_reprojection_error_after": float(rgb_error.mean().item()),
        "mean_feature_confidence_before": float(record.get("mean_feature_confidence", 0.0)),
        "mean_reprojection_confidence_after": float(c_reproj.mean().item()),
        "mean_feature_confidence_after": float(c_feat.mean().item()),
        "mean_soft_confidence_after": float(soft_confidence.mean().item()),
        "mean_hard_validity_after": float(hard_validity.mean().item()),
        "mean_visibility_confidence_after": float(c_vis.mean().item()),
        "mean_patch_weight_after": float(c_patch.mean().item()),
        "mean_final_mask_after": float(refreshed.mean().item()),
        "patch_keep_ratio_before": float(record.get("patch_keep_ratio", 1.0)),
        "patch_keep_ratio_after": float(patch_keep_ratio),
        "online_refresh_delta_confidence": float(refreshed.mean().item() - before.mean().item()),
        "depth_drift_vs_coarse": float(depth_drift),
    }
    if writeback_interval > 0 and iteration % writeback_interval == 0:
        safe_name = camera_name(viewpoint_cam).replace("/", "_")
        out = output_dir / "online_confidence" / f"{iteration:06d}_{safe_name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out) + "_mask.npy", refreshed[0].detach().cpu().numpy().astype(np.float32))
        save_tensor_image(Path(str(out) + "_mask.png"), refreshed[0])
        save_tensor_image(Path(str(out) + "_c_reproj.png"), c_reproj[0])
        save_tensor_image(Path(str(out) + "_c_feat.png"), c_feat[0])
        save_tensor_image(Path(str(out) + "_c_patch.png"), c_patch[0])
    return metrics


def log_training_diagnostics(
    *,
    tb_writer,
    output_dir: Path,
    iteration: int,
    viewpoint_cam,
    render_image: torch.Tensor,
    gt_image: torch.Tensor,
    render_depth: torch.Tensor | None,
    confidence_mask: torch.Tensor | None,
    record: dict | None,
    export_png: bool,
) -> None:
    safe_name = camera_name(viewpoint_cam).replace("/", "_")
    error = torch.abs(render_image.detach() - gt_image.detach()).mean(dim=0, keepdim=True)
    depth_vis = normalize_depth_for_drift(render_depth.detach()) if render_depth is not None else None
    if tb_writer:
        tb_writer.add_images(f"diagnostics/{safe_name}/render", render_image.detach()[None].clamp(0.0, 1.0), iteration)
        tb_writer.add_images(f"diagnostics/{safe_name}/target", gt_image.detach()[None].clamp(0.0, 1.0), iteration)
        tb_writer.add_images(f"diagnostics/{safe_name}/error_heatmap", error[None].clamp(0.0, 1.0), iteration)
        if depth_vis is not None:
            tb_writer.add_images(f"diagnostics/{safe_name}/inv_depth", depth_vis[None].clamp(0.0, 1.0), iteration)
        if confidence_mask is not None:
            tb_writer.add_images(f"diagnostics/{safe_name}/confidence", confidence_mask.detach()[None].clamp(0.0, 1.0), iteration)

    if not export_png:
        return
    prefix = output_dir / "diagnostics" / f"{iteration:06d}_{safe_name}"
    save_tensor_image(Path(str(prefix) + "_render.png"), render_image)
    save_tensor_image(Path(str(prefix) + "_target.png"), gt_image)
    save_tensor_image(Path(str(prefix) + "_error.png"), error[0])
    if depth_vis is not None:
        save_tensor_image(Path(str(prefix) + "_inv_depth.png"), depth_vis[0])
    if confidence_mask is not None:
        save_tensor_image(Path(str(prefix) + "_confidence.png"), confidence_mask[0])
    if record:
        for key, suffix in (
            ("c_reproj_path", "c_reproj"),
            ("c_feat_path", "c_feat"),
            ("c_patch_path", "c_patch"),
        ):
            component = load_numpy_mask(record.get(key), tuple(render_image.shape[-2:]), render_image.device)
            if component is not None:
                save_tensor_image(Path(str(prefix) + f"_{suffix}.png"), component[0])


def get_stage_settings(
    iteration: int,
    *,
    pseudo_warmup_iters: int,
    pseudo_full_iters: int,
    pseudo_ratio_mid: float,
    pseudo_ratio_final: float,
    pseudo_weight_mid: float,
    pseudo_weight_final: float,
) -> dict[str, float | bool | str]:
    if iteration < pseudo_warmup_iters:
        return {
            "name": "real_only_warmup",
            "use_pseudo": False,
            "pseudo_ratio": 0.0,
            "pseudo_weight": 0.0,
        }
    if iteration < pseudo_full_iters:
        return {
            "name": "hybrid_low_weight",
            "use_pseudo": True,
            "pseudo_ratio": float(pseudo_ratio_mid),
            "pseudo_weight": float(pseudo_weight_mid),
        }
    return {
        "name": "hybrid_full",
        "use_pseudo": True,
        "pseudo_ratio": float(pseudo_ratio_final),
        "pseudo_weight": float(pseudo_weight_final),
    }


def masked_lpips_loss(image, gt_image, mask, lpips_criterion, eps: float = 1e-6):
    pred = image
    target = gt_image
    norm = torch.ones((), device=image.device)
    if mask is not None:
        pred = pred * mask
        target = target * mask
        norm = mask.mean().clamp_min(eps)
    pred = pred.unsqueeze(0) * 2.0 - 1.0
    target = target.unsqueeze(0) * 2.0 - 1.0
    return lpips_criterion(pred, target).mean() / norm


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    confidence_from_alpha: bool,
    pseudo_warmup_iters: int,
    pseudo_full_iters: int,
    pseudo_ratio_mid: float,
    pseudo_ratio_final: float,
    pseudo_weight_mid: float,
    pseudo_weight_final: float,
    enable_pseudo_lpips: bool,
    pseudo_lpips_weight: float,
    confidence_manifest: str | None,
    enable_online_confidence: bool,
    confidence_refresh_interval: int,
    confidence_writeback_interval: int,
    online_rgb_sigma: float,
    online_feature_sigma: float,
    online_patch_size: int,
    online_patch_threshold: float,
    online_patch_low_weight: float,
    online_patch_min_keep_ratio: float,
    diagnostics_interval: int,
    diagnostics_debug_views: int,
    diagnostics_export_png: bool,
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, "
            "please install the correct rasterizer using pip install [3dgs_accel]."
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations,
    )

    real_cameras, pseudo_cameras = split_training_cameras(scene)
    confidence_records = load_confidence_records(confidence_manifest)
    diagnostic_cameras = select_diagnostic_cameras(real_cameras, pseudo_cameras, diagnostics_debug_views)
    output_dir = Path(dataset.model_path).expanduser()
    if confidence_records:
        print(f"Loaded {len(confidence_records)} confidence records for online refresh and sampling.")
    if diagnostic_cameras:
        print(
            "Diagnostics will track fixed views: "
            + ", ".join(camera_name(camera) for camera in diagnostic_cameras)
        )
    real_viewpoint_stack = real_cameras.copy()
    pseudo_viewpoint_stack = pseudo_cameras.copy()
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    lpips_criterion = None
    if enable_pseudo_lpips:
        if not LPIPS_AVAILABLE:
            print("LPIPS is not available in this environment. Continuing without pseudo LPIPS loss.")
        else:
            lpips_criterion = LPIPS(net_type="vgg").to("cuda").eval()
            for parameter in lpips_criterion.parameters():
                parameter.requires_grad = False

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                    net_image_bytes = memoryview(
                        (
                            torch.clamp(net_image, min=0, max=1.0)
                            * 255
                        )
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        stage_settings = get_stage_settings(
            iteration,
            pseudo_warmup_iters=pseudo_warmup_iters,
            pseudo_full_iters=pseudo_full_iters,
            pseudo_ratio_mid=pseudo_ratio_mid,
            pseudo_ratio_final=pseudo_ratio_final,
            pseudo_weight_mid=pseudo_weight_mid,
            pseudo_weight_final=pseudo_weight_final,
        )
        want_pseudo = bool(stage_settings["use_pseudo"]) and bool(pseudo_cameras) and random.random() < float(stage_settings["pseudo_ratio"])
        if want_pseudo:
            if confidence_records:
                viewpoint_cam = pop_weighted_pseudo_camera(pseudo_cameras, confidence_records)
            else:
                viewpoint_cam = pop_random_camera(pseudo_viewpoint_stack, pseudo_cameras)
        else:
            source_cameras = real_cameras if real_cameras else pseudo_cameras
            source_pool = real_viewpoint_stack if real_cameras else pseudo_viewpoint_stack
            viewpoint_cam = pop_random_camera(source_pool, source_cameras)
        camera_is_pseudo = is_pseudo_camera(viewpoint_cam)
        confidence_record = confidence_records.get(camera_name(viewpoint_cam), {})
        sampling_weight = float(confidence_record.get("clip_score", 1.0)) if camera_is_pseudo else 1.0
        pseudo_weight = float(stage_settings["pseudo_weight"]) if camera_is_pseudo else 1.0

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image_raw = viewpoint_cam.original_image.cuda()
        online_metrics = maybe_refresh_online_confidence(
            viewpoint_cam=viewpoint_cam,
            render_image=image,
            gt_image=gt_image_raw,
            render_depth=render_pkg.get("depth"),
            record=confidence_record if confidence_record else None,
            iteration=iteration,
            enabled=bool(camera_is_pseudo and enable_online_confidence),
            refresh_interval=int(confidence_refresh_interval),
            writeback_interval=int(confidence_writeback_interval),
            output_dir=output_dir,
            rgb_sigma=float(online_rgb_sigma),
            feature_sigma=float(online_feature_sigma),
            patch_size=int(online_patch_size),
            patch_threshold=float(online_patch_threshold),
            patch_low_weight=float(online_patch_low_weight),
            patch_min_keep_ratio=float(online_patch_min_keep_ratio),
        )
        gt_image = gt_image_raw
        confidence_mask = get_confidence_mask(viewpoint_cam, confidence_from_alpha)
        masked_image, masked_gt_image = apply_confidence_mask(image, gt_image, viewpoint_cam, confidence_from_alpha)
        mask_mean_value = float(confidence_mask.mean().item()) if confidence_mask is not None else 1.0
        effective_pseudo_weight = float(pseudo_weight) * mask_mean_value if camera_is_pseudo else 1.0
        legacy_effective_pseudo_weight = effective_pseudo_weight

        if camera_is_pseudo and confidence_mask is not None:
            Ll1 = masked_l1_loss(image, gt_image, confidence_mask)
            ssim_image = masked_image
            ssim_gt_image = masked_gt_image
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_image = image
            ssim_gt_image = gt_image
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(ssim_image.unsqueeze(0), ssim_gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(ssim_image, ssim_gt_image)
        base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        lpips_loss = torch.zeros((), device=image.device)
        if camera_is_pseudo and lpips_criterion is not None and pseudo_lpips_weight > 0:
            lpips_loss = masked_lpips_loss(image, gt_image, confidence_mask, lpips_criterion)
        if camera_is_pseudo:
            loss = pseudo_weight * (base_loss + float(pseudo_lpips_weight) * lpips_loss)
        else:
            loss = base_loss

        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            inv_depth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if tb_writer:
                tb_writer.add_scalar("train_schedule/pseudo_weight", float(stage_settings["pseudo_weight"]), iteration)
                tb_writer.add_scalar("train_schedule/pseudo_ratio", float(stage_settings["pseudo_ratio"]), iteration)
                tb_writer.add_scalar("train_schedule/is_pseudo_camera", float(camera_is_pseudo), iteration)
                tb_writer.add_scalar("train_schedule/pseudo_sampling_weight", float(sampling_weight), iteration)
                if camera_is_pseudo:
                    tb_writer.add_scalar("train_loss_patches/pseudo_loss", loss.item(), iteration)
                    tb_writer.add_scalar("train_schedule/pseudo_loss_weight", float(pseudo_weight), iteration)
                    if confidence_mask is not None:
                        tb_writer.add_scalar("confidence/mean_confidence", mask_mean_value, iteration)
                        tb_writer.add_scalar("train_schedule/effective_pseudo_weight", effective_pseudo_weight, iteration)
                        tb_writer.add_scalar("train_schedule/legacy_effective_pseudo_weight", legacy_effective_pseudo_weight, iteration)
                    for key in (
                        "mean_reprojection_error",
                        "mean_feature_confidence",
                        "mean_soft_confidence",
                        "mean_hard_validity",
                        "mean_padding_validity",
                        "mean_patch_weight",
                        "mean_final_mask",
                        "patch_keep_ratio",
                        "clip_score",
                    ):
                        if key in confidence_record:
                            tb_writer.add_scalar(f"confidence/{key}", float(confidence_record[key]), iteration)
                    for key, value in online_metrics.items():
                        tb_writer.add_scalar(f"online_confidence/{key}", float(value), iteration)
                else:
                    tb_writer.add_scalar("train_loss_patches/real_loss", loss.item(), iteration)
                if camera_is_pseudo and lpips_criterion is not None:
                    tb_writer.add_scalar("train_loss_patches/pseudo_lpips", lpips_loss.item(), iteration)

            if diagnostics_interval > 0 and iteration % int(diagnostics_interval) == 0:
                log_training_diagnostics(
                    tb_writer=tb_writer,
                    output_dir=output_dir,
                    iteration=iteration,
                    viewpoint_cam=viewpoint_cam,
                    render_image=render_pkg["render"],
                    gt_image=gt_image_raw,
                    render_depth=render_pkg.get("depth"),
                    confidence_mask=confidence_mask,
                    record=confidence_record if confidence_record else None,
                    export_png=bool(diagnostics_export_png),
                )
                logged_names = {camera_name(viewpoint_cam)}
                for diagnostic_cam in diagnostic_cameras:
                    if camera_name(diagnostic_cam) in logged_names:
                        continue
                    diagnostic_pkg = render(
                        diagnostic_cam,
                        gaussians,
                        pipe,
                        background,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )
                    diagnostic_record = confidence_records.get(camera_name(diagnostic_cam), {})
                    diagnostic_mask = get_confidence_mask(diagnostic_cam, confidence_from_alpha)
                    log_training_diagnostics(
                        tb_writer=tb_writer,
                        output_dir=output_dir,
                        iteration=iteration,
                        viewpoint_cam=diagnostic_cam,
                        render_image=diagnostic_pkg["render"],
                        gt_image=diagnostic_cam.original_image.cuda(),
                        render_depth=diagnostic_pkg.get("depth"),
                        confidence_mask=diagnostic_mask,
                        record=diagnostic_record if diagnostic_record else None,
                        export_png=bool(diagnostics_export_png),
                    )
                    logged_names.add(camera_name(diagnostic_cam))

            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.7f}",
                }
                if camera_is_pseudo:
                    postfix["mask_mean"] = f"{mask_mean_value:.4f}"
                    postfix["pseudo_w"] = f"{float(pseudo_weight):.4f}"
                    postfix["eff_w"] = f"{effective_pseudo_weight:.4f}"
                    postfix["legacy_eff_w"] = f"{legacy_effective_pseudo_weight:.4f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                confidence_from_alpha,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    render_func,
    render_args,
    train_test_exp,
    confidence_from_alpha: bool,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if not config["cameras"]:
                continue
            metrics = {
                "all_count": 0,
                "real_count": 0,
                "pseudo_count": 0,
                "full_l1_sum": 0.0,
                "full_psnr_sum": 0.0,
                "masked_unnorm_l1_sum": 0.0,
                "masked_unnorm_psnr_sum": 0.0,
                "masked_norm_l1_sum": 0.0,
                "masked_norm_psnr_sum": 0.0,
                "real_full_psnr_sum": 0.0,
                "pseudo_full_psnr_sum": 0.0,
            }
            for idx, viewpoint in enumerate(config["cameras"]):
                image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                mask = get_confidence_mask(viewpoint, confidence_from_alpha)
                masked_image, masked_gt_image = apply_confidence_mask(image, gt_image, viewpoint, confidence_from_alpha)
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2 :]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    masked_image = masked_image[..., masked_image.shape[-1] // 2 :]
                    masked_gt_image = masked_gt_image[..., masked_gt_image.shape[-1] // 2 :]
                    if mask is not None:
                        mask = mask[..., mask.shape[-1] // 2 :]
                if tb_writer and idx < 5:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_full", image[None], global_step=iteration)
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_masked_debug", masked_image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/ground_truth_full",
                            gt_image[None],
                            global_step=iteration,
                        )
                full_l1 = l1_loss_fn(image, gt_image).mean().double()
                full_psnr = psnr(image, gt_image).mean().double()
                masked_unnorm_l1 = l1_loss_fn(masked_image, masked_gt_image).mean().double()
                masked_unnorm_psnr = psnr(masked_image, masked_gt_image).mean().double()
                masked_norm_l1 = masked_l1_loss(image, gt_image, mask).double()
                masked_norm_psnr = psnr_from_mse(masked_mse(image, gt_image, mask, normalize=True)).double()

                metrics["all_count"] += 1
                metrics["full_l1_sum"] += float(full_l1.item())
                metrics["full_psnr_sum"] += float(full_psnr.item())
                metrics["masked_unnorm_l1_sum"] += float(masked_unnorm_l1.item())
                metrics["masked_unnorm_psnr_sum"] += float(masked_unnorm_psnr.item())
                metrics["masked_norm_l1_sum"] += float(masked_norm_l1.item())
                metrics["masked_norm_psnr_sum"] += float(masked_norm_psnr.item())
                if is_pseudo_camera(viewpoint):
                    metrics["pseudo_count"] += 1
                    metrics["pseudo_full_psnr_sum"] += float(full_psnr.item())
                else:
                    metrics["real_count"] += 1
                    metrics["real_full_psnr_sum"] += float(full_psnr.item())

            count = max(1, metrics["all_count"])
            full_l1 = metrics["full_l1_sum"] / count
            full_psnr = metrics["full_psnr_sum"] / count
            masked_unnorm_l1 = metrics["masked_unnorm_l1_sum"] / count
            masked_unnorm_psnr = metrics["masked_unnorm_psnr_sum"] / count
            masked_norm_l1 = metrics["masked_norm_l1_sum"] / count
            masked_norm_psnr = metrics["masked_norm_psnr_sum"] / count
            real_only_full_psnr = metrics["real_full_psnr_sum"] / metrics["real_count"] if metrics["real_count"] else None
            pseudo_only_full_psnr = metrics["pseudo_full_psnr_sum"] / metrics["pseudo_count"] if metrics["pseudo_count"] else None
            real_text = f"{real_only_full_psnr:.6f}" if real_only_full_psnr is not None else "n/a"
            pseudo_text = f"{pseudo_only_full_psnr:.6f}" if pseudo_only_full_psnr is not None else "n/a"
            print(
                f"\n[ITER {iteration}] Evaluating {config['name']}: "
                f"full_L1 {full_l1:.8f} full_PSNR {full_psnr:.6f} "
                f"masked_unnorm_L1_debug {masked_unnorm_l1:.8f} "
                f"masked_unnorm_PSNR_debug {masked_unnorm_psnr:.6f} "
                f"masked_norm_L1 {masked_norm_l1:.8f} masked_norm_PSNR {masked_norm_psnr:.6f} "
                f"real_only_full_PSNR {real_text} pseudo_only_full_PSNR {pseudo_text} "
                f"counts real={metrics['real_count']} pseudo={metrics['pseudo_count']}"
            )
            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/full_l1", full_l1, iteration)
                tb_writer.add_scalar(f"{config['name']}/full_psnr", full_psnr, iteration)
                tb_writer.add_scalar(f"{config['name']}/masked_unnorm_l1_debug", masked_unnorm_l1, iteration)
                tb_writer.add_scalar(f"{config['name']}/masked_unnorm_psnr_debug", masked_unnorm_psnr, iteration)
                tb_writer.add_scalar(f"{config['name']}/masked_norm_l1", masked_norm_l1, iteration)
                tb_writer.add_scalar(f"{config['name']}/masked_norm_psnr", masked_norm_psnr, iteration)
                tb_writer.add_scalar(f"{config['name']}/real_count", metrics["real_count"], iteration)
                tb_writer.add_scalar(f"{config['name']}/pseudo_count", metrics["pseudo_count"], iteration)
                if real_only_full_psnr is not None:
                    tb_writer.add_scalar(f"{config['name']}/real_only_full_psnr", real_only_full_psnr, iteration)
                if pseudo_only_full_psnr is not None:
                    tb_writer.add_scalar(f"{config['name']}/pseudo_only_full_psnr", pseudo_only_full_psnr, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def main() -> None:
    parser = ArgumentParser(description="Confidence-aware 3DGS training entrypoint for Part 3")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--confidence_from_alpha",
        action="store_true",
        default=False,
        help="Treat PNG alpha as a confidence mask and apply it to render/GT losses.",
    )
    parser.add_argument("--pseudo_warmup_iters", type=int, default=2000, help="Stage 1 ends here: train with only real sparse views.")
    parser.add_argument("--pseudo_full_iters", type=int, default=8000, help="Stage 2 ends here: switch to full hybrid sampling after this iteration.")
    parser.add_argument("--pseudo_ratio_mid", type=float, default=0.25, help="Pseudo-view sampling ratio during the low-weight hybrid stage.")
    parser.add_argument("--pseudo_ratio_final", type=float, default=0.5, help="Pseudo-view sampling ratio during the full hybrid stage.")
    parser.add_argument("--pseudo_weight_mid", type=float, default=0.2, help="Global pseudo-view loss weight during the low-weight hybrid stage.")
    parser.add_argument("--pseudo_weight_final", type=float, default=0.5, help="Global pseudo-view loss weight during the full hybrid stage.")
    parser.add_argument("--enable_pseudo_lpips", action="store_true", default=False, help="Add masked LPIPS on pseudo-view batches.")
    parser.add_argument("--pseudo_lpips_weight", type=float, default=0.05, help="LPIPS loss weight applied only on pseudo-view batches.")
    parser.add_argument("--confidence_manifest", type=str, default="", help="Part 3 confidence manifest with component maps and clip scores.")
    parser.add_argument("--enable_online_confidence", action="store_true", default=False, help="Refresh pseudo confidence masks during training.")
    parser.add_argument("--confidence_refresh_interval", type=int, default=200)
    parser.add_argument("--confidence_writeback_interval", type=int, default=1000)
    parser.add_argument("--online_rgb_sigma", type=float, default=0.2)
    parser.add_argument("--online_feature_sigma", type=float, default=0.25)
    parser.add_argument("--online_patch_size", type=int, default=16)
    parser.add_argument("--online_patch_threshold", type=float, default=0.25)
    parser.add_argument("--online_patch_low_weight", type=float, default=0.15)
    parser.add_argument("--online_patch_min_keep_ratio", type=float, default=0.1)
    parser.add_argument("--diagnostics_interval", type=int, default=1000)
    parser.add_argument("--diagnostics_debug_views", type=int, default=4)
    parser.add_argument("--disable_diagnostics_png", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.confidence_from_alpha,
        args.pseudo_warmup_iters,
        args.pseudo_full_iters,
        args.pseudo_ratio_mid,
        args.pseudo_ratio_final,
        args.pseudo_weight_mid,
        args.pseudo_weight_final,
        args.enable_pseudo_lpips,
        args.pseudo_lpips_weight,
        args.confidence_manifest or None,
        args.enable_online_confidence,
        args.confidence_refresh_interval,
        args.confidence_writeback_interval,
        args.online_rgb_sigma,
        args.online_feature_sigma,
        args.online_patch_size,
        args.online_patch_threshold,
        args.online_patch_low_weight,
        args.online_patch_min_keep_ratio,
        args.diagnostics_interval,
        args.diagnostics_debug_views,
        not args.disable_diagnostics_png,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
