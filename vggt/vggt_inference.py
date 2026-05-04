import os
import json
from pathlib import Path
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def run_vggt(image_dir, output_dir, chunk_size=4, conf_threshold=0.5):
    """
    分块运行 VGGT 推理，输出相机位姿和初始点云。

    Args:
        image_dir: 图像文件夹路径（支持 ~ 展开）
        output_dir: 输出文件夹路径（支持 ~ 展开）
        chunk_size: 每次处理的图像数量，显存不足时减小该值
        conf_threshold: 点云置信度阈值，用于过滤低质量点
    """
    image_dir = os.path.expanduser(image_dir)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print("device:", device, "dtype:", dtype)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"Total images: {len(image_files)}")

    # 用于合并所有块的输出
    all_pose_enc = []
    all_world_points = []
    all_world_points_conf = []
    all_extrinsics = []
    all_intrinsics = []

    for start_idx in range(0, len(image_files), chunk_size):
        end_idx = min(start_idx + chunk_size, len(image_files))
        chunk_paths = image_files[start_idx:end_idx]
        print(f"Processing chunk {start_idx//chunk_size + 1}/{(len(image_files)-1)//chunk_size + 1} "
              f"(images {start_idx+1} to {end_idx})")

        images = load_and_preprocess_images([str(p) for p in chunk_paths]).to(device)
        images = images.unsqueeze(0)  # (1, N, 3, H, W)
        _, _, H, W = images.shape[-4:]

        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype):
            preds = model(images)

        # 提取 pose_enc 并转换为相机内外参
        pose_enc = preds["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))

        # 收集数据
        all_pose_enc.append(pose_enc.cpu().numpy())
        all_extrinsics.append(extrinsic.cpu().numpy())
        all_intrinsics.append(intrinsic.cpu().numpy())
        all_world_points.append(preds["world_points"].cpu().numpy())
        all_world_points_conf.append(preds["world_points_conf"].cpu().numpy())

        del images, preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 沿序列维度合并所有块
    final_pose_enc = np.concatenate(all_pose_enc, axis=1)
    final_extrinsics = np.concatenate(all_extrinsics, axis=1)
    final_intrinsics = np.concatenate(all_intrinsics, axis=1)
    final_world_points = np.concatenate(all_world_points, axis=1)
    final_world_points_conf = np.concatenate(all_world_points_conf, axis=1)

    # 置信度过滤并保存点云为 PLY
    points = final_world_points[0]  # (N, H, W, 3)
    conf = final_world_points_conf[0]  # (N, H, W)
    mask = conf > conf_threshold
    filtered_points = points[mask]  # (M, 3)

    try:
        from plyfile import PlyData, PlyElement
        vertex = np.array([tuple(p) for p in filtered_points], 
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply = PlyData([PlyElement.describe(vertex, 'vertex')])
        ply.write(os.path.join(output_dir, "points.ply"))
        print(f"Saved PLY with {len(filtered_points)} points (threshold={conf_threshold})")
    except ImportError:
        print("plyfile not installed, skipping PLY export")
        np.save(os.path.join(output_dir, "points_filtered.npy"), filtered_points)

    # 保存完整输出
    np.save(os.path.join(output_dir, "pose_enc.npy"), final_pose_enc)
    np.save(os.path.join(output_dir, "extrinsics.npy"), final_extrinsics)
    np.save(os.path.join(output_dir, "intrinsics.npy"), final_intrinsics)
    np.save(os.path.join(output_dir, "world_points.npy"), final_world_points)
    np.save(os.path.join(output_dir, "world_points_conf.npy"), final_world_points_conf)

    # 保存元数据
    with open(os.path.join(output_dir, "vggt_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_images": len(image_files),
            "image_paths": [str(p) for p in image_files],
            "chunk_size": chunk_size,
            "device": device,
            "torch": torch.__version__,
            "conf_threshold": conf_threshold,
            "pose_enc_shape": list(final_pose_enc.shape),
            "extrinsics_shape": list(final_extrinsics.shape),
            "intrinsics_shape": list(final_intrinsics.shape),
        }, f, indent=2)

    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    run_vggt(
        "/path/to/your/CVproj/data/DL3DV-2/rgb",
        "/path/to/your/CVproj/results/vggt1",
        chunk_size=4,
        conf_threshold=0.5
    )
