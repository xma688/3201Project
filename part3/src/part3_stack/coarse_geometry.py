from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from .config import ProjectConfig, read_json, write_json


_COLMAP_IO: ModuleType | None = None


def _load_colmap_io(project_root: Path) -> ModuleType:
    global _COLMAP_IO
    if _COLMAP_IO is not None:
        return _COLMAP_IO

    module_path = project_root / "gaussian-splatting" / "utils" / "read_write_model.py"
    spec = importlib.util.spec_from_file_location("gs_read_write_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import COLMAP helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _COLMAP_IO = module
    return module


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for Part 3 geometry-guided refinement. "
            "Install opencv-python in the environment used for Part 3."
        ) from exc
    return cv2


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_rgb_to_shape(rgb: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if rgb.shape[:2] == (height, width):
        return rgb.astype(np.uint8, copy=False)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(out)
    return out


def _save_map_preview(path: str | Path, array: np.ndarray, *, invalid_value: float | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if invalid_value is not None:
        finite &= arr != invalid_value
    preview = np.zeros(arr.shape, dtype=np.uint8)
    if finite.any():
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
        if hi - lo < 1e-6:
            preview[finite] = 255
        else:
            preview[finite] = np.clip((arr[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(out)
    return out


def _save_alpha_preview(path: str | Path, alpha: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(out)
    return out


def camera_params_to_intrinsics(
    camera_model: str,
    camera_params: list[float] | np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    params = np.asarray(camera_params, dtype=np.float64)
    if camera_model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif camera_model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif camera_model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif camera_model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported camera model for Part 3 coarse rendering: {camera_model}")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(K).all():
        raise ValueError(f"Invalid intrinsics built from {camera_model=} {camera_params=}")
    return K


def qvec_tvec_to_pose_mats(project_root: Path, qvec: list[float] | np.ndarray, tvec: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colmap_io = _load_colmap_io(project_root)
    rot_cw = colmap_io.qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return rot_cw, t


def pose_mats_to_qvec_tvec(project_root: Path, rot_cw: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colmap_io = _load_colmap_io(project_root)
    qvec = colmap_io.rotmat2qvec(np.asarray(rot_cw, dtype=np.float64))
    return np.asarray(qvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64).reshape(3)


def camera_center_from_pose(rot_cw: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -rot_cw.T @ np.asarray(tvec, dtype=np.float64).reshape(3)


def world_to_camera(points_world: np.ndarray, rot_cw: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return points_world @ rot_cw.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)


def camera_to_world(points_cam: np.ndarray, rot_cw: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return (points_cam - np.asarray(tvec, dtype=np.float64).reshape(1, 3)) @ rot_cw


def project_camera_points(K: np.ndarray, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    xy = points_cam[:, :2] / np.maximum(z[:, None], 1e-8)
    uv = (xy @ K[:2, :2].T) + K[:2, 2]
    return uv, z


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (grid_x - cx) / fx
    y = (grid_y - cy) / fy
    xyz = np.stack([x * depth, y * depth, depth], axis=-1)
    return xyz


@dataclass(frozen=True)
class RenderBundle:
    rgb: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray
    normal: np.ndarray
    valid_mask: np.ndarray


class CoarseGeometryRenderer:
    def __init__(
        self,
        *,
        points: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray | None = None,
        splat_radius: int = 2,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.float32)
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float32)
        self.splat_radius = max(0, int(splat_radius))

    @classmethod
    def from_scene(
        cls,
        *,
        project_root: Path,
        base_scene_dir: str | Path,
        splat_radius: int = 2,
    ) -> "CoarseGeometryRenderer":
        colmap_io = _load_colmap_io(project_root)
        _cameras, _images, points3d = colmap_io.read_model(str(Path(base_scene_dir).expanduser() / "sparse" / "0"), ext="")
        if not points3d:
            raise RuntimeError(f"No COLMAP sparse points found under {base_scene_dir}")

        xyz = np.stack([np.asarray(point.xyz, dtype=np.float64) for point in points3d.values()], axis=0)
        rgb = np.stack([np.asarray(point.rgb, dtype=np.float32) / 255.0 for point in points3d.values()], axis=0)
        return cls(points=xyz, colors=rgb, normals=None, splat_radius=splat_radius)

    def render_view(
        self,
        *,
        project_root: Path,
        qvec: list[float] | np.ndarray,
        tvec: list[float] | np.ndarray,
        camera_model: str,
        camera_params: list[float] | np.ndarray,
        width: int,
        height: int,
    ) -> RenderBundle:
        K = camera_params_to_intrinsics(camera_model, camera_params, width, height)
        rot_cw, t = qvec_tvec_to_pose_mats(project_root, qvec, tvec)
        points_cam = world_to_camera(self.points, rot_cw, t)
        positive = points_cam[:, 2] > 1e-4
        if not np.any(positive):
            empty_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            empty_depth = np.full((height, width), np.inf, dtype=np.float32)
            empty_alpha = np.zeros((height, width), dtype=np.float32)
            empty_normal = np.zeros((height, width, 3), dtype=np.float32)
            empty_valid = np.zeros((height, width), dtype=bool)
            return RenderBundle(
                rgb=empty_rgb,
                depth=empty_depth,
                alpha=empty_alpha,
                normal=empty_normal,
                valid_mask=empty_valid,
            )

        points_cam = points_cam[positive]
        colors = self.colors[positive]
        normals = None if self.normals is None else self.normals[positive]
        uv, z = project_camera_points(K, points_cam)
        inside = (
            np.isfinite(z)
            & (z > 1e-4)
            & np.isfinite(uv[:, 0])
            & np.isfinite(uv[:, 1])
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        if not np.any(inside):
            empty_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            empty_depth = np.full((height, width), np.inf, dtype=np.float32)
            empty_alpha = np.zeros((height, width), dtype=np.float32)
            empty_normal = np.zeros((height, width, 3), dtype=np.float32)
            empty_valid = np.zeros((height, width), dtype=bool)
            return RenderBundle(
                rgb=empty_rgb,
                depth=empty_depth,
                alpha=empty_alpha,
                normal=empty_normal,
                valid_mask=empty_valid,
            )

        uv = uv[inside]
        z = z[inside]
        colors = colors[inside]
        normals = None if normals is None else normals[inside]

        rgb = np.zeros((height, width, 3), dtype=np.float32)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        normal_map = np.zeros((height, width, 3), dtype=np.float32)

        radius = self.splat_radius
        offsets = [(0, 0)] if radius <= 0 else [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        if radius > 0:
            sigma2 = max(1.0, float(radius * radius))
            weight_lut = {
                (dx, dy): float(np.exp(-0.5 * (dx * dx + dy * dy) / sigma2))
                for dx, dy in offsets
            }
        else:
            weight_lut = {(0, 0): 1.0}

        order = np.argsort(z)
        for idx in order:
            u = int(round(float(uv[idx, 0])))
            v = int(round(float(uv[idx, 1])))
            point_z = float(z[idx])
            color = colors[idx]
            normal = None if normals is None else normals[idx]
            for dx, dy in offsets:
                xx = u + dx
                yy = v + dy
                if xx < 0 or yy < 0 or xx >= width or yy >= height:
                    continue
                if point_z >= float(depth[yy, xx]):
                    continue
                depth[yy, xx] = point_z
                rgb[yy, xx] = color
                alpha[yy, xx] = weight_lut[(dx, dy)]
                if normal is not None:
                    normal_map[yy, xx] = normal

        valid_mask = np.isfinite(depth) & (alpha > 0)
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return RenderBundle(
            rgb=rgb_uint8,
            depth=depth,
            alpha=alpha,
            normal=normal_map.astype(np.float32),
            valid_mask=valid_mask,
        )


def save_render_bundle(bundle: RenderBundle, output_prefix: str | Path) -> dict[str, str]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = save_rgb_image(prefix.with_suffix(".png"), bundle.rgb)
    depth_path = prefix.parent / f"{prefix.name}_depth.npy"
    alpha_path = prefix.parent / f"{prefix.name}_alpha.npy"
    normal_path = prefix.parent / f"{prefix.name}_normal.npy"
    depth_preview_path = prefix.parent / f"{prefix.name}_depth.png"
    alpha_preview_path = prefix.parent / f"{prefix.name}_alpha.png"
    normal_preview_path = prefix.parent / f"{prefix.name}_normal.png"

    np.save(depth_path, bundle.depth.astype(np.float32))
    np.save(alpha_path, bundle.alpha.astype(np.float32))
    np.save(normal_path, bundle.normal.astype(np.float32))
    _save_map_preview(depth_preview_path, bundle.depth, invalid_value=np.inf)
    _save_alpha_preview(alpha_preview_path, bundle.alpha)

    normal_preview = np.clip((bundle.normal + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    save_rgb_image(normal_preview_path, normal_preview)
    return {
        "coarse_rgb_path": str(rgb_path),
        "coarse_depth_path": str(depth_path),
        "coarse_depth_preview_path": str(depth_preview_path),
        "coarse_alpha_path": str(alpha_path),
        "coarse_alpha_preview_path": str(alpha_preview_path),
        "coarse_normal_path": str(normal_path),
        "coarse_normal_preview_path": str(normal_preview_path),
        "coarse_valid_ratio": float(bundle.valid_mask.mean()),
    }


def load_render_bundle(view_record: dict[str, Any]) -> RenderBundle:
    rgb = load_rgb_image(view_record["coarse_rgb_path"])
    depth = np.load(view_record["coarse_depth_path"]).astype(np.float32)
    alpha = np.load(view_record["coarse_alpha_path"]).astype(np.float32)
    normal = np.load(view_record["coarse_normal_path"]).astype(np.float32)
    valid_mask = np.isfinite(depth) & (alpha > 0)
    return RenderBundle(rgb=rgb, depth=depth, alpha=alpha, normal=normal, valid_mask=valid_mask)


def _view_camera_meta(view_record: dict[str, Any]) -> tuple[str, list[float], int, int]:
    return (
        str(view_record["camera_model"]),
        [float(v) for v in view_record["camera_params"]],
        int(view_record["width"]),
        int(view_record["height"]),
    )


def render_trajectory_coarse_guidance(
    config: ProjectConfig,
    trajectory_manifest_path: str | Path,
    *,
    splat_radius: int,
) -> Path:
    trajectory = read_json(trajectory_manifest_path)
    run_dir = Path(trajectory["run_dir"]).expanduser()
    coarse_root = run_dir / "coarse_geometry"
    renderer = CoarseGeometryRenderer.from_scene(
        project_root=config.project_root,
        base_scene_dir=trajectory["base_scene_dir"],
        splat_radius=splat_radius,
    )

    for pair in trajectory["pairs"]:
        pair_dir = coarse_root / pair["pair_id"]
        for view_record in pair["intermediate_views"]:
            camera_model, camera_params, width, height = _view_camera_meta(view_record)
            bundle = renderer.render_view(
                project_root=config.project_root,
                qvec=view_record["qvec"],
                tvec=view_record["tvec"],
                camera_model=camera_model,
                camera_params=camera_params,
                width=width,
                height=height,
            )
            out_prefix = pair_dir / Path(view_record["output_name"]).stem
            view_record.update(save_render_bundle(bundle, out_prefix))

    trajectory["coarse_guidance_root"] = str(coarse_root)
    return write_json(trajectory_manifest_path, trajectory)


def build_visibility_confidence(
    depth: np.ndarray,
    alpha: np.ndarray,
    *,
    alpha_threshold: float,
    boundary_percentile: float,
    boundary_dilate: int,
) -> np.ndarray:
    cv2 = _require_cv2()
    valid = (alpha > float(alpha_threshold)) & np.isfinite(depth)
    if not valid.any():
        return np.zeros_like(alpha, dtype=np.float32)

    depth_for_grad = depth.copy()
    if valid.any():
        fill_value = float(np.median(depth[valid]))
        depth_for_grad[~valid] = fill_value
    grad_x = cv2.Sobel(depth_for_grad.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_for_grad.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    threshold = float(np.percentile(grad_mag[valid], boundary_percentile)) if valid.any() else np.inf
    boundary = grad_mag >= threshold
    kernel_size = max(1, int(boundary_dilate))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    vis = valid & (~boundary)
    return vis.astype(np.float32)


def warp_source_to_target(
    *,
    project_root: Path,
    source_rgb: np.ndarray,
    source_qvec: list[float] | np.ndarray,
    source_tvec: list[float] | np.ndarray,
    source_camera_model: str,
    source_camera_params: list[float] | np.ndarray,
    target_qvec: list[float] | np.ndarray,
    target_tvec: list[float] | np.ndarray,
    target_camera_model: str,
    target_camera_params: list[float] | np.ndarray,
    target_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    height, width = target_depth.shape
    target_K = camera_params_to_intrinsics(target_camera_model, target_camera_params, width, height)
    source_K = camera_params_to_intrinsics(source_camera_model, source_camera_params, source_rgb.shape[1], source_rgb.shape[0])
    target_rot, target_t = qvec_tvec_to_pose_mats(project_root, target_qvec, target_tvec)
    source_rot, source_t = qvec_tvec_to_pose_mats(project_root, source_qvec, source_tvec)

    valid_target = np.isfinite(target_depth) & (target_depth > 1e-4)
    safe_depth = target_depth.astype(np.float64).copy()
    safe_depth[~valid_target] = 0.0
    target_points_cam = backproject_depth(safe_depth, target_K)
    target_points_world = camera_to_world(target_points_cam.reshape(-1, 3), target_rot, target_t)
    source_points_cam = world_to_camera(target_points_world, source_rot, source_t)
    source_uv, source_z = project_camera_points(source_K, source_points_cam)
    source_uv = source_uv.reshape(height, width, 2).astype(np.float32)
    source_z = source_z.reshape(height, width)

    in_bounds = (
        valid_target
        & (source_z > 1e-4)
        & np.isfinite(source_uv[..., 0])
        & np.isfinite(source_uv[..., 1])
        & (source_uv[..., 0] >= 0)
        & (source_uv[..., 0] <= source_rgb.shape[1] - 1)
        & (source_uv[..., 1] >= 0)
        & (source_uv[..., 1] <= source_rgb.shape[0] - 1)
    )
    warped = cv2.remap(
        source_rgb.astype(np.float32),
        source_uv[..., 0],
        source_uv[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.float32), in_bounds


def compute_anchor_reprojection(
    *,
    project_root: Path,
    pseudo_rgb: np.ndarray,
    target_qvec: list[float] | np.ndarray,
    target_tvec: list[float] | np.ndarray,
    target_camera_model: str,
    target_camera_params: list[float] | np.ndarray,
    target_depth: np.ndarray,
    start_anchor_path: str | Path,
    start_qvec: list[float] | np.ndarray,
    start_tvec: list[float] | np.ndarray,
    start_camera_model: str,
    start_camera_params: list[float] | np.ndarray,
    end_anchor_path: str | Path,
    end_qvec: list[float] | np.ndarray,
    end_tvec: list[float] | np.ndarray,
    end_camera_model: str,
    end_camera_params: list[float] | np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    start_rgb = load_rgb_image(start_anchor_path)
    end_rgb = load_rgb_image(end_anchor_path)

    warped_start, valid_start = warp_source_to_target(
        project_root=project_root,
        source_rgb=start_rgb,
        source_qvec=start_qvec,
        source_tvec=start_tvec,
        source_camera_model=start_camera_model,
        source_camera_params=start_camera_params,
        target_qvec=target_qvec,
        target_tvec=target_tvec,
        target_camera_model=target_camera_model,
        target_camera_params=target_camera_params,
        target_depth=target_depth,
    )
    warped_end, valid_end = warp_source_to_target(
        project_root=project_root,
        source_rgb=end_rgb,
        source_qvec=end_qvec,
        source_tvec=end_tvec,
        source_camera_model=end_camera_model,
        source_camera_params=end_camera_params,
        target_qvec=target_qvec,
        target_tvec=target_tvec,
        target_camera_model=target_camera_model,
        target_camera_params=target_camera_params,
        target_depth=target_depth,
    )

    weight_start = (1.0 - float(alpha)) * valid_start.astype(np.float32)
    weight_end = float(alpha) * valid_end.astype(np.float32)
    denom = weight_start + weight_end
    blend = np.zeros_like(warped_start, dtype=np.float32)
    blend += warped_start * weight_start[..., None]
    blend += warped_end * weight_end[..., None]
    valid = denom > 1e-6
    if valid.any():
        blend[valid] /= denom[valid][..., None]

    pseudo_rgb = resize_rgb_to_shape(pseudo_rgb, target_depth.shape)
    pseudo_float = pseudo_rgb.astype(np.float32)
    error_map = np.ones(target_depth.shape, dtype=np.float32)
    if valid.any():
        error_map[valid] = np.abs(pseudo_float[valid] - blend[valid]).mean(axis=1) / 255.0
    mean_error = float(error_map[valid].mean()) if valid.any() else 1.0
    return {
        "blend_rgb": np.clip(blend, 0, 255).astype(np.uint8),
        "valid_mask": valid.astype(np.float32),
        "error_map": error_map.astype(np.float32),
        "mean_error": mean_error,
    }


def refine_pose_with_pnp(
    *,
    project_root: Path,
    pseudo_rgb: np.ndarray,
    coarse_bundle: RenderBundle,
    init_qvec: list[float] | np.ndarray,
    init_tvec: list[float] | np.ndarray,
    camera_model: str,
    camera_params: list[float] | np.ndarray,
    min_matches: int = 20,
) -> dict[str, Any]:
    cv2 = _require_cv2()
    gray_pseudo = cv2.cvtColor(pseudo_rgb, cv2.COLOR_RGB2GRAY)
    gray_coarse = cv2.cvtColor(coarse_bundle.rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=2048)
    kp_pseudo, des_pseudo = orb.detectAndCompute(gray_pseudo, None)
    kp_coarse, des_coarse = orb.detectAndCompute(gray_coarse, None)
    if des_pseudo is None or des_coarse is None or not kp_pseudo or not kp_coarse:
        return {
            "success": False,
            "qvec": list(np.asarray(init_qvec, dtype=np.float64)),
            "tvec": list(np.asarray(init_tvec, dtype=np.float64)),
            "num_matches": 0,
            "num_inliers": 0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_pseudo, des_coarse), key=lambda item: item.distance)
    K = camera_params_to_intrinsics(camera_model, camera_params, pseudo_rgb.shape[1], pseudo_rgb.shape[0])
    rot_init, t_init = qvec_tvec_to_pose_mats(project_root, init_qvec, init_tvec)

    pts3d: list[np.ndarray] = []
    pts2d: list[tuple[float, float]] = []
    for match in matches[:512]:
        u_coarse, v_coarse = kp_coarse[match.trainIdx].pt
        uu = int(round(float(u_coarse)))
        vv = int(round(float(v_coarse)))
        if uu < 0 or vv < 0 or uu >= coarse_bundle.depth.shape[1] or vv >= coarse_bundle.depth.shape[0]:
            continue
        if coarse_bundle.alpha[vv, uu] <= 1e-4:
            continue
        z = float(coarse_bundle.depth[vv, uu])
        if not np.isfinite(z) or z <= 1e-4:
            continue
        x_cam = np.array(
            [
                (u_coarse - K[0, 2]) / K[0, 0] * z,
                (v_coarse - K[1, 2]) / K[1, 1] * z,
                z,
            ],
            dtype=np.float64,
        )
        x_world = (x_cam - t_init) @ rot_init
        pts3d.append(x_world)
        pts2d.append(kp_pseudo[match.queryIdx].pt)

    if len(pts3d) < max(12, int(min_matches)):
        return {
            "success": False,
            "qvec": list(np.asarray(init_qvec, dtype=np.float64)),
            "tvec": list(np.asarray(init_tvec, dtype=np.float64)),
            "num_matches": len(pts3d),
            "num_inliers": 0,
        }

    pts3d_np = np.asarray(pts3d, dtype=np.float64)
    pts2d_np = np.asarray(pts2d, dtype=np.float64)
    rvec_init, _ = cv2.Rodrigues(rot_init)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d_np,
        pts2d_np,
        K,
        None,
        rvec=rvec_init,
        tvec=t_init.reshape(3, 1),
        useExtrinsicGuess=True,
        iterationsCount=100,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {
            "success": False,
            "qvec": list(np.asarray(init_qvec, dtype=np.float64)),
            "tvec": list(np.asarray(init_tvec, dtype=np.float64)),
            "num_matches": len(pts3d),
            "num_inliers": 0,
        }

    try:
        rvec, tvec = cv2.solvePnPRefineLM(pts3d_np, pts2d_np, K, None, rvec, tvec)
    except Exception:
        pass

    rot_refined, _ = cv2.Rodrigues(rvec)
    qvec_refined, tvec_refined = pose_mats_to_qvec_tvec(project_root, rot_refined, tvec.reshape(3))
    return {
        "success": True,
        "qvec": qvec_refined.tolist(),
        "tvec": tvec_refined.tolist(),
        "num_matches": len(pts3d),
        "num_inliers": int(0 if inliers is None else len(inliers)),
    }


def score_pseudo_frame(
    *,
    pseudo_rgb: np.ndarray,
    coarse_rgb: np.ndarray,
    reprojection_error: float,
    coarse_l1_weight: float = 0.6,
    reprojection_weight: float = 0.4,
) -> dict[str, float]:
    pseudo_rgb = resize_rgb_to_shape(pseudo_rgb, coarse_rgb.shape[:2])
    coarse_l1 = float(np.abs(pseudo_rgb.astype(np.float32) - coarse_rgb.astype(np.float32)).mean() / 255.0)
    frame_score = coarse_l1_weight * coarse_l1 + reprojection_weight * float(reprojection_error)
    return {
        "coarse_l1": coarse_l1,
        "reprojection_error": float(reprojection_error),
        "frame_score": float(frame_score),
    }
