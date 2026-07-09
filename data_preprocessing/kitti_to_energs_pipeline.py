#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI MOT -> 3DGS Dataset + Energy Field (All-in-One Pipeline)

Pipeline stages (identical to the Waymo pipeline):
  1. Read images and LiDAR point clouds for the requested frame range.
  2. Export camera intrinsics/extrinsics in COLMAP text format.
  3. Crop the point cloud to a Camera-Union bounding box along the trajectory.
  4. Voxel down-sample and denoise (Statistical Outlier Removal) the LiDAR.
  5. Cast rays from an extended frame window to build the FREE volume.
  6. Derive OCC/FREE/UNK masks and save them as a compact energy field.
  7. Write a run_log.json capturing the exact arguments and per-phase stats.

Expected KITTI-MOT directory layout:
  <kitti_root>/
    image_02/<seq_id>/*.png     # left  colour camera
    image_03/<seq_id>/*.png     # right colour camera (optional)
    velodyne/<seq_id>/*.bin     # LiDAR points in the Velodyne frame
    calib/<seq_id>.txt          # P2, P3, R_rect, Tr_velo_cam, Tr_imu_velo
    oxts/<seq_id>.txt           # per-frame IMU/GPS pose

Example:
  python kitti_to_energs_pipeline.py \
      --kitti_root /path/to/kitti_mot/training \
      --seq_id 0001 \
      --out_root  /path/to/output_root \
      --dataset_start_frame 30 --dataset_end_frame 50 \
      --free_backward_frames 20 --free_forward_frames 20 \
      --voxel_size 0.1 --field_voxel_size 0.5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from math import cos, pi

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

# Try to import numba for JIT acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARN] numba not available, using pure numpy (slower)")


# ============================================================
# KITTI Calibration & Pose Parsing
# ============================================================

def kitti_string_to_float(s: str) -> float:
    s = s.strip()
    if 'e' in s or 'E' in s:
        base, expo = s.lower().split('e')
        return float(base) * 10 ** int(expo)
    return float(s)


def parse_kitti_calib(calib_path: Path) -> dict:
    """Parse KITTI tracking calibration file."""
    lines = calib_path.read_text().splitlines()
    calibs = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        vals = [kitti_string_to_float(v) for v in parts[1:]]
        calibs.append(np.array(vals, dtype=np.float64))

    if len(calibs) < 7:
        raise RuntimeError(f"Unexpected calib format in {calib_path}")

    P0 = calibs[0].reshape(3, 4)
    P1 = calibs[1].reshape(3, 4)
    P2 = calibs[2].reshape(3, 4)
    P3 = calibs[3].reshape(3, 4)

    R_rect = calibs[4].reshape(3, 3)
    Tr_cam2camrect = np.eye(4, dtype=np.float64)
    Tr_cam2camrect[:3, :3] = R_rect

    Tr_velo2cam = calibs[5].reshape(3, 4)
    Tr_velo2cam = np.vstack([Tr_velo2cam, np.array([0.0, 0.0, 0.0, 1.0])])

    Tr_imu2velo = calibs[6].reshape(3, 4)
    Tr_imu2velo = np.vstack([Tr_imu2velo, np.array([0.0, 0.0, 0.0, 1.0])])

    return {
        "P0": P0, "P1": P1, "P2": P2, "P3": P3,
        "Tr_cam2camrect": Tr_cam2camrect,
        "Tr_velo2cam": Tr_velo2cam,
        "Tr_imu2velo": Tr_imu2velo,
    }


def get_rotation(roll, pitch, heading):
    """roll, pitch, heading (yaw) -> 3x3 rotation matrix"""
    s_heading, c_heading = np.sin(heading), np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])
    s_pitch, c_pitch = np.sin(pitch), np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])
    s_roll, c_roll = np.sin(roll), np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    return rot_z @ (rot_y @ rot_x)


def oxts_to_poses(oxts: np.ndarray) -> np.ndarray:
    """Convert OXTS data to IMU poses (world <- IMU)."""
    def latlon_to_mercator(lat, lon, scale):
        r = 6378137.0
        x = scale * r * (pi * lon / 180.0)
        y = scale * r * np.log(np.tan((pi * (90.0 + lat)) / 360.0))
        return x, y

    N = oxts.shape[0]
    poses = []
    lat0 = oxts[0, 0]
    scale = cos(lat0 * pi / 180.0)
    pose_0_inv = None

    for i in range(N):
        vals = oxts[i]
        lat, lon, alt = vals[0], vals[1], vals[2]
        roll, pitch, yaw = vals[3], vals[4], vals[5]
        x, y = latlon_to_mercator(lat, lon, scale)
        t = np.array([x, y, alt], dtype=np.float64)
        R = get_rotation(roll, pitch, yaw)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3], T[:3, 3] = R, t
        if pose_0_inv is None:
            pose_0_inv = invert_T(T)
        poses.append(pose_0_inv @ T)
    return np.stack(poses, axis=0)


def load_oxts_poses(oxts_path: Path) -> np.ndarray:
    """Load OXTS file and return IMU poses."""
    arr = np.loadtxt(str(oxts_path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return oxts_to_poses(arr)


# ============================================================
# Transform helpers
# ============================================================

def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ t
    return Ti


def rotmat_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        qw, qx, qy, qz = (m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s
    elif m[1,1] > m[2,2]:
        s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        qw, qx, qy, qz = (m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s
    else:
        s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        qw, qx, qy, qz = (m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ============================================================
# Point cloud processing
# ============================================================

def read_velodyne_bin(bin_path: Path) -> np.ndarray:
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size % 4 != 0:
        arr = arr[: (arr.size // 4) * 4]
    return arr.reshape(-1, 4)[:, :3]  # only xyz


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    print(f"[VOXEL] Input points: {pts.shape[0]}, voxel_size: {voxel_size}m")
    xyz = pts[:, :3].astype(np.float64)
    
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    voxel_min = voxel_idx.min(axis=0)
    voxel_idx = voxel_idx - voxel_min
    voxel_max = voxel_idx.max(axis=0) + 1
    
    print(f"[VOXEL] Grid size: {voxel_max[0]} x {voxel_max[1]} x {voxel_max[2]}")
    
    total_voxels = int(voxel_max[0]) * int(voxel_max[1]) * int(voxel_max[2])
    if total_voxels > 2**62:
        print(f"[VOXEL] WARNING: Grid too large, using lexsort")
        order = np.lexsort((voxel_idx[:, 2], voxel_idx[:, 1], voxel_idx[:, 0]))
        sorted_idx = voxel_idx[order]
        diff = np.any(sorted_idx[1:] != sorted_idx[:-1], axis=1)
        unique_mask = np.concatenate([[True], diff])
        unique_idx = order[unique_mask]
    else:
        flat_idx = voxel_idx[:, 0] * (voxel_max[1] * voxel_max[2]) + voxel_idx[:, 1] * voxel_max[2] + voxel_idx[:, 2]
        _, unique_idx = np.unique(flat_idx, return_index=True)
    
    result = pts[unique_idx]
    print(f"[VOXEL] Output points: {result.shape[0]} (reduction: {100*(1 - result.shape[0]/pts.shape[0]):.1f}%)")
    return result


def statistical_outlier_removal(pts: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Remove outliers using Statistical Outlier Removal (SOR)."""
    if pts.shape[0] < nb_neighbors + 1:
        return pts
    xyz = pts[:, :3].astype(np.float64)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=nb_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances <= threshold
    return pts[mask]


def crop_points_by_camera_union(
    pts: np.ndarray,
    cam_centers: np.ndarray,
    forward_extend: float = 80.0,
    backward_extend: float = 10.0,
    lateral_half_width: float = 30.0,
    height_above_ground: float = 40.0,
) -> Tuple[np.ndarray, dict]:
    """Crop points by union of per-camera oriented boxes."""
    if pts.shape[0] == 0:
        return pts, {}
    
    xyz = pts[:, :3].astype(np.float64)
    cams = cam_centers.astype(np.float64)
    n_cams = cams.shape[0]
    
    # Per-camera forward direction
    forward_vecs = np.zeros((n_cams, 2), dtype=np.float64)
    for i in range(n_cams):
        if i < n_cams - 1:
            fwd = cams[i + 1, :2] - cams[i, :2]
        else:
            fwd = cams[i, :2] - cams[i - 1, :2] if n_cams > 1 else np.array([1.0, 0.0])
        fwd_norm = np.linalg.norm(fwd)
        forward_vecs[i] = fwd / fwd_norm if fwd_norm > 1e-6 else np.array([1.0, 0.0])
    
    pts_xy = xyz[:, :2]
    inside_any = np.zeros(xyz.shape[0], dtype=np.bool_)
    
    for i in range(n_cams):
        cam_xy = cams[i, :2]
        fwd = forward_vecs[i]
        lat = np.array([-fwd[1], fwd[0]])
        rel = pts_xy - cam_xy
        pts_fwd = rel[:, 0] * fwd[0] + rel[:, 1] * fwd[1]
        pts_lat = rel[:, 0] * lat[0] + rel[:, 1] * lat[1]
        in_box = (pts_fwd >= -backward_extend) & (pts_fwd <= forward_extend) & \
                 (pts_lat >= -lateral_half_width) & (pts_lat <= lateral_half_width)
        inside_any |= in_box
    
    ground_z = xyz[inside_any, 2].min() if inside_any.sum() > 0 else xyz[:, 2].min()
    z_min, z_max = ground_z, ground_z + height_above_ground
    mask = inside_any & (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    
    overall_fwd = cams[-1] - cams[0]
    overall_fwd[2] = 0
    overall_fwd_norm = np.linalg.norm(overall_fwd)
    overall_fwd = overall_fwd / overall_fwd_norm if overall_fwd_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    overall_lat = np.array([-overall_fwd[1], overall_fwd[0], 0.0])
    traj_center = cams.mean(axis=0)
    
    cam_fwd_proj = np.dot(cams[:, :2] - traj_center[:2], overall_fwd[:2])
    box_forward_min = cam_fwd_proj.min() - backward_extend
    box_forward_max = cam_fwd_proj.max() + forward_extend
    
    box_info = {
        "traj_center": traj_center, "forward_vec": overall_fwd, "lateral_vec": overall_lat,
        "box_forward_range": (box_forward_min, box_forward_max),
        "box_lateral_range": (-lateral_half_width, lateral_half_width),
        "ground_z": ground_z, "box_height_range": (z_min, z_max),
        "forward_length": box_forward_max - box_forward_min,
        "lateral_width": 2 * lateral_half_width, "height": height_above_ground,
    }
    return pts[mask], box_info


# ============================================================
# COLMAP writers
# ============================================================

def export_cameras_txt(f, width, height, fx, fy, cx, cy):
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write("# Number of cameras: 1\n")
    f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def export_images_txt(f, image_data: List[Tuple[np.ndarray, np.ndarray, str]]):
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(image_data)}\n")
    for image_id, (R_wc, C_world, name) in enumerate(image_data, start=1):
        R_cw = R_wc.T
        t_cw = -R_cw @ C_world.reshape(3)
        qw, qx, qy, qz = rotmat_to_quat_wxyz(R_cw)
        f.write(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} 1 {name}\n\n")


def export_points3D_ply(points_xyz: np.ndarray, ply_path: Path):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.size == 0:
        ply_path.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        return
    xyz = pts[:, :3].astype(np.float32)
    n = xyz.shape[0]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    data = np.zeros(n, dtype=dtype)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['red'], data['green'], data['blue'] = 200, 200, 200
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with ply_path.open("wb") as f:
        f.write(header.encode('ascii'))
        data.tofile(f)


# ============================================================
# Field generation (DDA ray traversal)
# ============================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def dda_ray_numba(cam_center, direction, origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth):
        X, Y, Z = dims[0], dims[1], dims[2]
        d = direction.copy()
        dn = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
        if dn < 1e-12:
            return
        d = d / dn
        p = cam_center.copy()
        v0 = np.floor((p - origin) / voxel_size).astype(np.int32)
        ix, iy, iz = v0[0], v0[1], v0[2]
        if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z):
            return
        step = np.zeros(3, dtype=np.int32)
        for ax in range(3):
            step[ax] = 1 if d[ax] >= 0 else -1
        tMax, tDelta = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        for ax in range(3):
            if abs(d[ax]) < 1e-12:
                tMax[ax], tDelta[ax] = 1e30, 1e30
            else:
                boundary = origin[ax] + (v0[ax] + (1 if step[ax] > 0 else 0)) * voxel_size
                tMax[ax] = (boundary - p[ax]) / d[ax]
                tDelta[ax] = voxel_size / abs(d[ax])
        traveled, max_keep = 0.0, min(free_max_depth, max_dist)
        max_keep_steps = int(max_keep / voxel_size)
        written_count = 0
        written_x = np.zeros(10000, dtype=np.int32)
        written_y = np.zeros(10000, dtype=np.int32)
        written_z = np.zeros(10000, dtype=np.int32)
        while traveled < max_dist:
            if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z) or occ_mask[ix, iy, iz]:
                break
            depth_steps = int(traveled / voxel_size)
            if depth_steps <= max_keep_steps:
                if depth_steps < free_depth[ix, iy, iz]:
                    free_depth[ix, iy, iz] = depth_steps
                if written_count < 10000:
                    written_x[written_count] = ix
                    written_y[written_count] = iy
                    written_z[written_count] = iz
                    written_count += 1
            ax = 0
            if tMax[1] < tMax[ax]: ax = 1
            if tMax[2] < tMax[ax]: ax = 2
            traveled = tMax[ax]
            if ax == 0: ix += step[0]; tMax[0] += tDelta[0]
            elif ax == 1: iy += step[1]; tMax[1] += tDelta[1]
            else: iz += step[2]; tMax[2] += tDelta[2]
            if traveled > max_keep:
                break
        if hit_guard > 0 and written_count > 0:
            k = int(np.ceil(hit_guard / voxel_size))
            if k > 0 and written_count > k:
                for i in range(written_count - k, written_count):
                    free_depth[written_x[i], written_y[i], written_z[i]] = 65535

    @njit(parallel=True, cache=True)
    def process_rays_batch_numba(cam_centers, directions, origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth):
        N = cam_centers.shape[0]
        for i in prange(N):
            dda_ray_numba(cam_centers[i], directions[i], origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth)
else:
    def dda_ray_numpy(cam_center, direction, origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth):
        X, Y, Z = dims[0], dims[1], dims[2]
        d = direction.copy()
        dn = np.sqrt(np.sum(d * d))
        if dn < 1e-12:
            return
        d = d / dn
        p = cam_center.copy()
        v0 = np.floor((p - origin) / voxel_size).astype(np.int32)
        ix, iy, iz = int(v0[0]), int(v0[1]), int(v0[2])
        if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z):
            return
        step = np.array([1 if d[ax] >= 0 else -1 for ax in range(3)], dtype=np.int32)
        tMax = np.zeros(3, dtype=np.float64)
        tDelta = np.zeros(3, dtype=np.float64)
        for ax in range(3):
            if abs(d[ax]) < 1e-12:
                tMax[ax], tDelta[ax] = 1e30, 1e30
            else:
                boundary = origin[ax] + (v0[ax] + (1 if step[ax] > 0 else 0)) * voxel_size
                tMax[ax] = (boundary - p[ax]) / d[ax]
                tDelta[ax] = voxel_size / abs(d[ax])
        traveled = 0.0
        max_keep = min(free_max_depth, max_dist)
        max_keep_steps = int(max_keep / voxel_size)
        written = []
        while traveled < max_dist:
            if not (0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z) or occ_mask[ix, iy, iz]:
                break
            depth_steps = int(traveled / voxel_size)
            if depth_steps <= max_keep_steps:
                if depth_steps < free_depth[ix, iy, iz]:
                    free_depth[ix, iy, iz] = depth_steps
                written.append((ix, iy, iz))
            ax = np.argmin(tMax)
            traveled = tMax[ax]
            if ax == 0: ix += step[0]; tMax[0] += tDelta[0]
            elif ax == 1: iy += step[1]; tMax[1] += tDelta[1]
            else: iz += step[2]; tMax[2] += tDelta[2]
            if traveled > max_keep:
                break
        if hit_guard > 0 and len(written) > 0:
            k = int(np.ceil(hit_guard / voxel_size))
            if k > 0 and len(written) > k:
                for (wx, wy, wz) in written[-k:]:
                    free_depth[wx, wy, wz] = 65535

    def process_rays_batch_numba(cam_centers, directions, origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth):
        print("[WARN] Using slow numpy fallback")
        for i in range(cam_centers.shape[0]):
            dda_ray_numpy(cam_centers[i], directions[i], origin, dims, voxel_size, occ_mask, free_depth, max_dist, hit_guard, free_max_depth)


def dilate_6n(mask: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return mask.copy()
    cur = mask.copy()
    for _ in range(steps):
        xp, xm = np.zeros_like(cur), np.zeros_like(cur)
        yp, ym = np.zeros_like(cur), np.zeros_like(cur)
        zp, zm = np.zeros_like(cur), np.zeros_like(cur)
        xp[1:], xm[:-1] = cur[:-1], cur[1:]
        yp[:, 1:], ym[:, :-1] = cur[:, :-1], cur[:, 1:]
        zp[:, :, 1:], zm[:, :, :-1] = cur[:, :, :-1], cur[:, :, 1:]
        cur = cur | xp | xm | yp | ym | zp | zm
    return cur


def save_field_npz(path, occ_mask, free_mask, unk_mask, voxel_size, grid_origin, roi_min, dims, meta):
    shape = occ_mask.shape
    occ_p = np.packbits(occ_mask.reshape(-1), bitorder="little")
    free_p = np.packbits(free_mask.reshape(-1), bitorder="little")
    unk_p = np.packbits(unk_mask.reshape(-1), bitorder="little")
    np.savez_compressed(path, occ_p=occ_p, free_p=free_p, unk_p=unk_p,
                        shape=np.array(shape, dtype=np.int32), voxel_size=float(voxel_size),
                        grid_origin=np.asarray(grid_origin, dtype=np.float64),
                        roi_min=np.asarray(roi_min, dtype=np.int32),
                        dims=np.asarray(dims, dtype=np.int32), format="field_cache_v1_packbits_little", **meta)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(args):
    kitti_root = Path(args.kitti_root)
    seq_id = args.seq_id
    out_root = Path(args.out_root)
    out_scene_dir = out_root / seq_id
    
    # KITTI paths
    img_dir = kitti_root / "image_02" / seq_id
    velo_dir = kitti_root / "velodyne" / seq_id
    calib_path = kitti_root / "calib" / f"{seq_id}.txt"
    oxts_path = kitti_root / "oxts" / f"{seq_id}.txt"
    
    # Output paths
    images_root = out_scene_dir / "images"
    sparse_root = out_scene_dir / "sparse" / "0"
    images_root.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize log
    run_log = {
        "start_time": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "stats": {},
    }
    
    # Validate inputs
    for p, name in [(img_dir, "image_02"), (velo_dir, "velodyne"), (calib_path, "calib"), (oxts_path, "oxts")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    
    # Load calibration and poses
    calib = parse_kitti_calib(calib_path)
    poses_imu_w = load_oxts_poses(oxts_path)
    
    # Camera intrinsics from P2
    P2 = calib["P2"]
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    
    # Transform chains
    Tr_imu2velo = calib["Tr_imu2velo"]
    Tr_velo2imu = invert_T(Tr_imu2velo)
    Tr_velo2cam = calib["Tr_velo2cam"]
    Tr_cam2velo = invert_T(Tr_velo2cam)
    
    # poses_velo_w: world <- velo
    poses_velo_w = poses_imu_w @ Tr_velo2imu
    
    # Get frame counts
    img_files = sorted(img_dir.glob("*.png"))
    velo_files = sorted(velo_dir.glob("*.bin"))
    n_frames_total = min(len(img_files), len(velo_files), poses_imu_w.shape[0])
    
    # Frame ranges
    dataset_start = max(0, args.dataset_start_frame)
    dataset_end = min(n_frames_total, args.dataset_end_frame)
    free_start = max(0, dataset_start - args.free_backward_frames)
    free_end = min(n_frames_total, dataset_end + args.free_forward_frames)
    
    print(f"[CONFIG] Sequence: {seq_id}")
    print(f"[CONFIG] Total frames: {n_frames_total}")
    print(f"[CONFIG] Dataset frame range: [{dataset_start}, {dataset_end})")
    print(f"[CONFIG] Freespace frame range: [{free_start}, {free_end})")
    print(f"[CONFIG] Point cloud voxel: {args.voxel_size}m, Field voxel: {args.field_voxel_size}m")
    
    # Get first image size
    im0 = Image.open(img_files[0])
    W, H = im0.size
    im0.close()
    
    # Storage
    image_data = []
    dataset_cam_centers = []
    selected_cam_instances = []
    freespace_cam_instances = []
    first_frame_cam_centers = []
    all_points_world = []
    
    # ========== Phase 1: Read KITTI data ==========
    print("\n[PHASE 1] Reading KITTI data...")
    for frame_idx in tqdm(range(n_frames_total), desc="Processing frames"):
        in_dataset_range = dataset_start <= frame_idx < dataset_end
        in_free_range = free_start <= frame_idx < free_end
        frame_id = f"{frame_idx:06d}"
        
        # Camera pose: world <- cam
        T_w_velo = poses_velo_w[frame_idx]
        T_w_cam = T_w_velo @ Tr_cam2velo
        R_wc = T_w_cam[:3, :3]
        C_world = T_w_cam[:3, 3].copy()
        
        # For dataset: save images and camera params
        if in_dataset_range:
            # Copy image
            src_img = img_files[frame_idx]
            dst_img = images_root / f"{frame_id}.png"
            if not dst_img.exists():
                dst_img.write_bytes(src_img.read_bytes())
            
            image_data.append((R_wc, C_world, f"{frame_id}.png"))
            dataset_cam_centers.append(C_world.reshape(1, 3))
            T_cam_world = invert_T(T_w_cam)
            selected_cam_instances.append((W, H, fx, fy, cx, cy, T_cam_world, R_wc, C_world))
            
            if frame_idx == dataset_start:
                first_frame_cam_centers.append(C_world.copy())
        
        # For freespace
        if in_free_range:
            T_cam_world = invert_T(T_w_cam)
            freespace_cam_instances.append((W, H, fx, fy, cx, cy, T_cam_world, R_wc, C_world))
        
        # LiDAR points
        if frame_idx < len(velo_files):
            pts_velo = read_velodyne_bin(velo_files[frame_idx])
            if pts_velo.shape[0] > 0:
                pts_velo_h = np.hstack([pts_velo, np.ones((pts_velo.shape[0], 1), dtype=np.float32)])
                pts_world = (T_w_velo @ pts_velo_h.T).T[:, :3]
                all_points_world.append(pts_world.astype(np.float32))
    
    print(f"[INFO] Dataset cameras: {len(selected_cam_instances)}, Freespace cameras: {len(freespace_cam_instances)}")
    
    run_log["stats"]["total_frames"] = n_frames_total
    run_log["stats"]["dataset_cameras"] = len(selected_cam_instances)
    run_log["stats"]["freespace_cameras"] = len(freespace_cam_instances)
    
    # ========== Phase 2: Point Cloud Processing ==========
    print("\n[PHASE 2] Processing point cloud...")
    pts_all = np.concatenate(all_points_world, axis=0) if all_points_world else np.zeros((0, 3), dtype=np.float32)
    print(f"[INFO] Total LiDAR points: {pts_all.shape[0]}")
    run_log["stats"]["lidar_points_raw"] = int(pts_all.shape[0])
    
    # 1. Crop by camera union
    if dataset_cam_centers:
        cam_centers_arr = np.concatenate(dataset_cam_centers, axis=0)
        pts_cropped, box_info = crop_points_by_camera_union(pts_all, cam_centers_arr, args.forward_extend, args.backward_extend, args.lateral_half_width, args.height_above_ground)
        print(f"[INFO] Points after crop: {pts_cropped.shape[0]}")
    else:
        pts_cropped, box_info = pts_all, {}
    run_log["stats"]["lidar_points_after_crop"] = int(pts_cropped.shape[0])
    
    # 2. Voxel downsample
    pts_downsampled = voxel_downsample(pts_cropped, args.voxel_size) if pts_cropped.shape[0] > 0 else pts_cropped
    print(f"[INFO] Points after voxel downsample: {pts_downsampled.shape[0]}")
    run_log["stats"]["lidar_points_after_downsample"] = int(pts_downsampled.shape[0])
    
    # 3. Statistical Outlier Removal
    if args.sor_neighbors > 0 and pts_downsampled.shape[0] > 0:
        pts_before = pts_downsampled.shape[0]
        pts_downsampled = statistical_outlier_removal(pts_downsampled, nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std_ratio)
        print(f"[DENOISE] SOR (k={args.sor_neighbors}, std={args.sor_std_ratio}): {pts_before} -> {pts_downsampled.shape[0]} points")
    
    # ========== Phase 3: Export COLMAP format ==========
    print("\n[PHASE 3] Exporting COLMAP format...")
    export_points3D_ply(pts_downsampled, sparse_root / "points3D.ply")
    
    with (sparse_root / "cameras.txt").open("w") as f:
        export_cameras_txt(f, W, H, fx, fy, cx, cy)
    
    with (sparse_root / "images.txt").open("w") as f:
        export_images_txt(f, image_data)
    
    print(f"[OK] Exported COLMAP to {sparse_root}")
    
    # ========== Phase 4: Generate Energy Field ==========
    print("\n[PHASE 4] Generating energy field...")
    if pts_downsampled.shape[0] == 0 or not box_info:
        print("[WARN] No data for field generation")
        return
    
    voxel = float(args.field_voxel_size)
    pts = pts_downsampled[:, :3].astype(np.float64)
    
    # Grid from point cloud bounds
    ground_z = box_info["ground_z"]
    height = box_info["height"]
    pts_xy_min = pts[:, :2].min(axis=0)
    pts_xy_max = pts[:, :2].max(axis=0)
    margin = voxel * 2
    grid_min = np.array([pts_xy_min[0] - margin, pts_xy_min[1] - margin, ground_z - margin])
    grid_max = np.array([pts_xy_max[0] + margin, pts_xy_max[1] + margin, ground_z + height + margin])
    grid_origin = grid_min.copy()
    dims = np.ceil((grid_max - grid_min) / voxel).astype(np.int32)
    X, Y, Z = int(dims[0]), int(dims[1]), int(dims[2])
    print(f"[ROI] voxel={voxel}m, dims=[{X},{Y},{Z}], total={X*Y*Z/1e6:.2f}M voxels")
    
    # OCC mask
    occ_idx = np.floor((pts - grid_origin) / voxel).astype(np.int32)
    valid = (occ_idx[:, 0] >= 0) & (occ_idx[:, 0] < X) & (occ_idx[:, 1] >= 0) & (occ_idx[:, 1] < Y) & (occ_idx[:, 2] >= 0) & (occ_idx[:, 2] < Z)
    occ_mask = np.zeros((X, Y, Z), dtype=np.bool_)
    occ_idx_valid = occ_idx[valid]
    occ_mask[occ_idx_valid[:, 0], occ_idx_valid[:, 1], occ_idx_valid[:, 2]] = True
    print(f"[OCC] {int(occ_mask.sum())} voxels")
    
    # LiDAR height for FREE cutoff
    lidar_max_z = np.percentile(pts[:, 2], 99)
    lidar_max_z_idx = int(np.floor((lidar_max_z - grid_origin[2]) / voxel))
    lidar_max_z_idx = min(max(lidar_max_z_idx, 0), Z - 1)
    print(f"[HEIGHT] LiDAR max z (99%): {lidar_max_z:.2f}m, grid z-index: {lidar_max_z_idx}/{Z}")
    
    # FREE by ray traversal
    free_depth = np.full((X, Y, Z), 65535, dtype=np.uint16)
    dims_arr = np.array([X, Y, Z], dtype=np.int32)
    origin_arr = grid_origin.astype(np.float64)
    free_max_depth = float(args.free_max_depth)
    
    print(f"[FREE] Processing {len(freespace_cam_instances)} cameras...")
    for (W, H, fx, fy, cx, cy, T_cam_world, R_wc, C_w) in tqdm(freespace_cam_instances, desc="Ray traversal"):
        us, vs = np.arange(0, W, args.pixel_stride), np.arange(0, H, args.pixel_stride)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.reshape(-1), vv.reshape(-1)
        if len(uu) > args.max_rays_per_pose:
            sel = np.random.choice(len(uu), args.max_rays_per_pose, replace=False)
            uu, vv = uu[sel], vv[sel]
        dirs_cam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=1)
        dirs_w = (R_wc @ dirs_cam.T).T
        dirs_w /= (np.linalg.norm(dirs_w, axis=1, keepdims=True) + 1e-12)
        centers = np.tile(C_w, (dirs_w.shape[0], 1))
        process_rays_batch_numba(centers.astype(np.float64), dirs_w.astype(np.float64), origin_arr, dims_arr, voxel, occ_mask, free_depth, free_max_depth, args.hit_guard, free_max_depth)
    
    max_keep_steps = int(free_max_depth / voxel)
    free_mask = (free_depth != 65535) & (free_depth <= max_keep_steps)
    print(f"[FREE] Before height cutoff: {int(free_mask.sum())} voxels")
    
    if lidar_max_z_idx < Z - 1:
        free_mask[:, :, lidar_max_z_idx + 1:] = False
    print(f"[FREE] After height cutoff: {int(free_mask.sum())} voxels")
    
    # UNK generation
    near_occ_steps = int(np.ceil(args.near_occ_depth / voxel))
    near_free_steps = int(np.ceil(args.near_free_depth / voxel))
    
    if near_occ_steps > 0:
        near_occ = dilate_6n(occ_mask, near_occ_steps)
        free_cutoff_count = int((free_mask & near_occ & (~occ_mask)).sum())
        free_mask[near_occ & (~occ_mask)] = False
        print(f"[FREE] Cut {free_cutoff_count} voxels near OCC surface")
        unk_near_occ = near_occ & (~occ_mask)
    else:
        unk_near_occ = np.zeros_like(occ_mask)
    
    if near_free_steps > 0:
        near_free = dilate_6n(free_mask, near_free_steps)
        unk_near_free = near_free & (~occ_mask) & (~free_mask)
    else:
        unk_near_free = np.zeros_like(free_mask)
    
    unk_mask = unk_near_occ | unk_near_free
    print(f"[UNK] Near OCC: {int(unk_near_occ.sum())}, Near FREE: {int((unk_near_free & ~unk_near_occ).sum())} voxels")
    
    # Ceiling UNK
    ceiling_layers = int(args.ceiling_unk_layers)
    if ceiling_layers > 0:
        has_content = occ_mask | free_mask
        has_content_any = has_content.any(axis=2)
        z_indices = np.arange(Z)[np.newaxis, np.newaxis, :]
        content_z_max = np.where(has_content, z_indices, -1).max(axis=2)
        ceiling_unk_before = int(unk_mask.sum())
        for layer in range(1, ceiling_layers + 1):
            target_z = content_z_max + layer
            valid_mask = has_content_any & (target_z >= 0) & (target_z < Z)
            xs, ys = np.where(valid_mask)
            zs = target_z[valid_mask].astype(np.int64)
            not_occ_at_target = ~occ_mask[xs, ys, zs]
            unk_mask[xs[not_occ_at_target], ys[not_occ_at_target], zs[not_occ_at_target]] = True
        print(f"[UNK] Ceiling: {ceiling_layers} layers, added {int(unk_mask.sum()) - ceiling_unk_before} voxels")
    
    # Rear cutoff
    if len(first_frame_cam_centers) > 0 and len(dataset_cam_centers) > 1:
        first_cams = np.array(first_frame_cam_centers)
        all_cams = np.concatenate(dataset_cam_centers, axis=0)
        forward_vec = all_cams[-1] - all_cams[0]
        forward_vec[2] = 0
        forward_norm = np.linalg.norm(forward_vec)
        if forward_norm > 1e-6:
            forward_vec = forward_vec / forward_norm
            first_cam_center = first_cams.mean(axis=0)
            first_cam_fwd_proj = np.dot(first_cams[:, :2] - first_cam_center[:2], forward_vec[:2])
            rear_cutoff_offset = first_cam_fwd_proj.min()
            voxel_coords_x = grid_origin[0] + (np.arange(X) + 0.5) * voxel
            voxel_coords_y = grid_origin[1] + (np.arange(Y) + 0.5) * voxel
            vx, vy = np.meshgrid(voxel_coords_x, voxel_coords_y, indexing='ij')
            voxel_fwd_proj = (vx - first_cam_center[0]) * forward_vec[0] + (vy - first_cam_center[1]) * forward_vec[1]
            rear_margin = args.backward_extend
            rear_mask_2d = voxel_fwd_proj < (rear_cutoff_offset - rear_margin)
            rear_mask_3d = np.broadcast_to(rear_mask_2d[:, :, np.newaxis], (X, Y, Z))
            occ_rear = int((occ_mask & rear_mask_3d).sum())
            free_rear = int((free_mask & rear_mask_3d).sum())
            unk_rear = int((unk_mask & rear_mask_3d).sum())
            occ_mask[rear_mask_3d] = False
            free_mask[rear_mask_3d] = False
            unk_mask[rear_mask_3d] = False
            print(f"[REAR CUTOFF] Removed {occ_rear} OCC, {free_rear} FREE, {unk_rear} UNK")
    
    print(f"[FINAL] OCC: {int(occ_mask.sum())}, FREE: {int(free_mask.sum())}, UNK: {int(unk_mask.sum())} voxels")
    
    # Log field stats
    run_log["stats"]["field_grid_dims"] = [X, Y, Z]
    run_log["stats"]["field_voxel_size"] = voxel
    run_log["stats"]["field_occ_voxels"] = int(occ_mask.sum())
    run_log["stats"]["field_free_voxels"] = int(free_mask.sum())
    run_log["stats"]["field_unk_voxels"] = int(unk_mask.sum())
    
    # Save field
    field_out_path = out_scene_dir / args.field_out
    roi_min = np.array([0, 0, 0], dtype=np.int32)
    meta = dict(scene=str(out_scene_dir), free_max_depth=free_max_depth, near_occ_depth=args.near_occ_depth, near_free_depth=args.near_free_depth)
    save_field_npz(str(field_out_path), occ_mask, free_mask, unk_mask, voxel, grid_origin, roi_min, dims, meta)
    print(f"[SAVE] Field saved to {field_out_path}")
    
    # Save run log
    run_log["end_time"] = datetime.now().isoformat()
    run_log["stats"]["lidar_points_final"] = int(pts_downsampled.shape[0])
    log_path = out_scene_dir / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"[SAVE] Run log saved to {log_path}")
    print("[DONE]")


def parse_args():
    p = argparse.ArgumentParser(description="KITTI MOT -> 3DGS Dataset + Energy Field")
    p.add_argument("--kitti_root", type=str, required=True, help="KITTI MOT training root")
    p.add_argument("--seq_id", type=str, required=True, help="Sequence ID (e.g., 0001)")
    p.add_argument("--out_root", type=str, required=True, help="Output root directory")
    p.add_argument("--dataset_start_frame", type=int, default=0)
    p.add_argument("--dataset_end_frame", type=int, default=50)
    p.add_argument("--free_backward_frames", type=int, default=20)
    p.add_argument("--free_forward_frames", type=int, default=20)
    p.add_argument("--forward_extend", type=float, default=50.0)
    p.add_argument("--backward_extend", type=float, default=0.0)
    p.add_argument("--lateral_half_width", type=float, default=30.0)
    p.add_argument("--height_above_ground", type=float, default=40.0)
    p.add_argument("--ceiling_unk_layers", type=int, default=3)
    p.add_argument("--sor_neighbors", type=int, default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--field_voxel_size", type=float, default=0.5)
    p.add_argument("--free_max_depth", type=float, default=60.0)
    p.add_argument("--pixel_stride", type=int, default=8)
    p.add_argument("--max_rays_per_pose", type=int, default=40000)
    p.add_argument("--hit_guard", type=float, default=0.75)
    p.add_argument("--near_occ_depth", type=float, default=1.0)
    p.add_argument("--near_free_depth", type=float, default=1.0)
    p.add_argument("--field_out", type=str, default="field_cache.npz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
