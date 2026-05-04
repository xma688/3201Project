#!/usr/bin/env python3
"""
Convert VGGT output (.npy files) to COLMAP sparse reconstruction format.
This script reads the pose_enc, extrinsics, intrinsics, world_points and
world_points_conf files saved by vggt_inference.py and writes the standard
cameras.txt, images.txt, and points3D.txt files for COLMAP.

Usage:
    python vggt_to_colmap.py --input_dir /path/to/vggt/output --output_dir /path/to/colmap/sparse
"""

import os
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement


def rotation_matrix_to_quaternion(matrix):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    r = R.from_matrix(matrix)
    quat = r.as_quat()  # Returns [x, y, z, w] in scipy
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]


def write_cameras_txt(filepath, intrinsics, image_size):
    """
    Write cameras.txt file.
    VGGT uses intrinsic matrices, we'll treat as PINHOLE model.
    """
    n_cameras = len(intrinsics)
    with open(filepath, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {n_cameras}\n")

        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed IDs
            width, height = image_size
            # VGGT intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            # If fx and fy are close, use SIMPLE_PINHOLE; otherwise PINHOLE
            if abs(fx - fy) < 1e-4:
                model = "SIMPLE_PINHOLE"
                params = [fx, cx, cy]
            else:
                model = "PINHOLE"
                params = [fx, fy, cx, cy]

            param_str = " ".join(f"{p:.6f}" for p in params)
            f.write(f"{camera_id} {model} {width} {height} {param_str}\n")

    print(f"Written {filepath} ({n_cameras} cameras)")


def write_images_txt(filepath, extrinsics, image_files):
    """
    Write images.txt file.
    COLMAP uses quaternion + translation, camera center = -R^T * t.
    """
    n_images = len(extrinsics)
    with open(filepath, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n_images}, mean observations per image: 0\n\n")

        for i, extrinsic in enumerate(extrinsics):
            image_id = i + 1
            camera_id = 1  # Assuming all images share same camera

            # VGGT extrinsic: [R|t], where R is 3x3 rotation, t is 3x1 translation
            # This is world-to-camera: p_cam = R * p_world + t
            R_mat = extrinsic[:3, :3]
            t_vec = extrinsic[:3, 3]

            # COLMAP expects quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
            # with camera center = -R^T * t
            quat = rotation_matrix_to_quaternion(R_mat)
            qw, qx, qy, qz = quat

            # In COLMAP, translation is the camera center in world coordinates
            # camera_center = -R^T * t
            camera_center = -R_mat.T @ t_vec
            tx, ty, tz = camera_center

            # Get image name
            img_name = Path(image_files[i]).name

            # Write image line
            f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                    f"{tx:.6f} {ty:.6f} {tz:.6f} {camera_id} {img_name}\n")
            # Write points line (empty, no 2D-3D correspondences)
            f.write("\n")

    print(f"Written {filepath} ({n_images} images)")


def write_points3d_txt(filepath, world_points, world_points_conf, conf_threshold=0.5):
    """Write points3D.txt file."""
    points = world_points[0]  # (N, H, W, 3)
    conf = world_points_conf[0]  # (N, H, W)
    mask = conf > conf_threshold
    filtered_points = points[mask]  # (M, 3)

    with open(filepath, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(filtered_points)}, mean track length: 0\n")

        for i, point in enumerate(filtered_points):
            point_id = i + 1
            x, y, z = point
            # Default gray color (no RGB from VGGT)
            r, g, b = 128, 128, 128
            error = 0.0
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error}\n")

    print(f"Written {filepath} ({len(filtered_points)} 3D points)")
    return filtered_points


def write_points_ply(filepath, world_points, world_points_conf, conf_threshold=0.5):
    """Write filtered points as PLY file with RGB colors (from original images)."""
    points = world_points[0]  # (N, H, W, 3)
    conf = world_points_conf[0]  # (N, H, W)
    mask = conf > conf_threshold
    filtered_points = points[mask]  # (M, 3)

    vertex = np.array(
        [(x, y, z) for x, y, z in filtered_points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    ply = PlyData([PlyElement.describe(vertex, 'vertex')])
    ply.write(filepath)
    print(f"Written {filepath} ({len(filtered_points)} points)")


def main():
    parser = argparse.ArgumentParser(description="Convert VGGT output to COLMAP format")
    parser.add_argument("--input_dir", "-i", required=True,
                        help="Directory containing VGGT .npy files")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Output directory for COLMAP sparse model")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.5,
                        help="Confidence threshold for point cloud filtering")
    parser.add_argument("--image_size", "-s", type=int, nargs=2, default=None,
                        help="Image width and height (e.g., --image_size 1920 1080)")
    parser.add_argument("--write_ply", action="store_true",
                        help="Also write filtered points as PLY file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VGGT outputs
    extrinsics = np.load(input_dir / "extrinsics.npy")[0]  # (N, 4, 4)
    intrinsics = np.load(input_dir / "intrinsics.npy")[0]  # (N, 3, 3)
    world_points = np.load(input_dir / "world_points.npy")
    world_points_conf = np.load(input_dir / "world_points_conf.npy")

    # Load meta data for image paths
    import json
    with open(input_dir / "vggt_meta.json", "r") as f:
        meta = json.load(f)
    image_files = meta["image_paths"]

    # Determine image size
    if args.image_size is not None:
        width, height = args.image_size
        image_size = (width, height)
    else:
        # Try to infer from intrinsics shape or meta
        if "extrinsics_shape" in meta:
            _, n_images, _, _ = meta["extrinsics_shape"]
        else:
            n_images = len(image_files)

        # Get image size from first image if possible, otherwise use default
        try:
            from PIL import Image
            first_img = Image.open(image_files[0])
            width, height = first_img.size
            image_size = (width, height)
        except (ImportError, FileNotFoundError):
            image_size = (1920, 1080)  # Default
            print(f"Warning: Could not determine image size, using default {image_size}")

    # Write COLMAP format files
    write_cameras_txt(output_dir / "cameras.txt", intrinsics, image_size)
    write_images_txt(output_dir / "images.txt", extrinsics, image_files)
    filtered_points = write_points3d_txt(output_dir / "points3D.txt",
                                         world_points, world_points_conf,
                                         args.conf_threshold)

    if args.write_ply:
        write_points_ply(output_dir / "points.ply",
                        world_points, world_points_conf,
                        args.conf_threshold)

    print(f"\nConversion complete! COLMAP sparse model saved to {output_dir}")
    print("You can load this directory in COLMAP GUI via File > Import model")


if __name__ == "__main__":
    main()