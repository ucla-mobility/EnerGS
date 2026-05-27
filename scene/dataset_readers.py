# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
from pathlib import Path
from typing import NamedTuple, List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


# -----------------------------
# Data structs
# -----------------------------
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


# -----------------------------
# Utils
# -----------------------------
def getNerfppNorm(cam_info: List[CameraInfo]) -> Dict[str, np.ndarray]:
    def get_center_and_diag(cam_centers: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = float(np.max(dist))
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


def _list_sparse_subdirs(scene_root: str) -> List[str]:
    sparse_root = os.path.join(scene_root, "sparse")
    if not os.path.isdir(sparse_root):
        return []
    subdirs = []
    for d in sorted(os.listdir(sparse_root)):
        full = os.path.join(sparse_root, d)
        if os.path.isdir(full):
            subdirs.append(d)
    return subdirs


def _read_colmap_pair(sdir: str):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sdir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sdir, "cameras.bin"))
        return cam_extrinsics, cam_intrinsics
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sdir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sdir, "cameras.txt"))
        return cam_extrinsics, cam_intrinsics


def _make_unique_uid(sid: str, intr_id: int) -> int:
    """
    Unique & stable uid across sparse subdirs.
    """
    try:
        s = int(sid)
        return s * 100000 + int(intr_id)
    except Exception:
        h = 0
        for ch in sid:
            h = (h * 131 + ord(ch)) % 100000
        return h * 100000 + int(intr_id)


def fetchPly(path: str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    if 'nx' in vertices.data.dtype.names:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray):
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]

    normals = np.zeros_like(xyz, dtype=np.float32)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz.astype(np.float32), normals, rgb.astype(np.uint8)), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def _find_points3d_base(path: str, sparse_ids: List[str]) -> Optional[str]:
    """
    Find a sparse/<sid> that contains points3D.*
    """
    for sid in sparse_ids:
        sdir = os.path.join(path, "sparse", sid)
        if os.path.exists(os.path.join(sdir, "points3D.ply")):
            return sdir
    for sid in sparse_ids:
        sdir = os.path.join(path, "sparse", sid)
        if os.path.exists(os.path.join(sdir, "points3D.bin")) or os.path.exists(os.path.join(sdir, "points3D.txt")):
            return sdir
    return None


def _normalize_relpath(p: str) -> str:
    return p.replace("\\", "/").strip("/")


def _resolve_image_path(scene_root: str, reading_dir: str, sid: str, extr_name: str) -> str:
    """
    For your structure:
      - images/<sid>/<file>
    Also supports extr.name already includes "sid/file" (avoid double sid).
    Also supports flat images/<file> as fallback.

    extr_name examples:
      - "000047.png"
      - "2/000047.png"
    """
    base = os.path.join(scene_root, reading_dir)
    rel = _normalize_relpath(extr_name)
    rel_os = rel.replace("/", os.path.sep)

    # If extr already includes a subdir, use it directly: images/<rel>
    if "/" in rel:
        return os.path.join(base, rel_os)

    # Otherwise prefer images/<sid>/<file>
    cand = os.path.join(base, str(sid), rel_os)
    if os.path.exists(cand):
        return cand

    # Fallback: images/<file>
    return os.path.join(base, rel_os)


def _resolve_depth_path(scene_root: str, depths_dir: str, sid: str, extr_name: str) -> str:
    """
    Depth structure can mirror images:
      - depths/<sid>/<stem>.png
      - depths/<stem>.png
      - if extr_name already includes "sid/..." -> depths/<sid>/<stem>.png by direct join
    """
    if depths_dir == "":
        return ""

    base = os.path.join(scene_root, depths_dir)
    rel = _normalize_relpath(extr_name)
    stem = os.path.splitext(rel)[0]  # may include "sid/000047"

    if "/" in stem:
        return os.path.join(base, stem.replace("/", os.path.sep) + ".png")

    cand = os.path.join(base, str(sid), stem + ".png")
    if os.path.exists(cand):
        return cand

    return os.path.join(base, stem + ".png")


def _extract_frame_key_from_image_name(image_name: str) -> str:
    """
    image_name examples (after we prefix):
      - "s2_000047.png"
      - "s2_2/000047.png"
    Returns frame key "000047" (string).
    """
    # remove "s{sid}_" prefix
    rest = image_name.split("_", 1)[1] if "_" in image_name else image_name
    rest = _normalize_relpath(rest)         # "2/000047.png" or "000047.png"
    rest = rest.split("/")[-1]              # "000047.png"
    stem = os.path.splitext(rest)[0]        # "000047"
    return stem


# -----------------------------
# COLMAP camera reading (per sparse/<sid>)
# -----------------------------
def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    scene_root: str,
    reading_dir: str,
    depths_dir: str,
    sid: str,
    name_prefix: str,
    test_name_set: set,
) -> List[CameraInfo]:
    cam_infos: List[CameraInfo] = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write(f"Reading camera {idx+1}/{len(cam_extrinsics)} (sparse/{sid})")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = _make_unique_uid(sid, intr.id)

        # COLMAP qvec/tvec: world->cam
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            fx = intr.params[0]
            FovY = focal2fov(fx, height)
            FovX = focal2fov(fx, width)
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)
        else:
            raise AssertionError(
                "Colmap camera model not handled: only undistorted datasets "
                "(PINHOLE or SIMPLE_PINHOLE) supported!"
            )

        # Depth params lookup (optional)
        depth_params = None
        if depths_params is not None:
            name_noext = os.path.splitext(_normalize_relpath(extr.name).split("/")[-1])[0]
            try:
                depth_params = depths_params[name_noext]
            except Exception:
                depth_params = None

        # ---- Unique image_name (avoid collisions across cameras) ----
        raw = _normalize_relpath(extr.name)  # "000047.png" or "2/000047.png"
        image_name = f"{name_prefix}{raw}"

        # ---- Resolve actual file paths robustly (fix double sid) ----
        image_path = _resolve_image_path(scene_root, reading_dir, sid, extr.name)
        depth_path = _resolve_depth_path(scene_root, depths_dir, sid, extr.name) if depths_dir != "" else ""

        is_test = (image_name in test_name_set)

        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params=depth_params,
                image_path=image_path,
                image_name=image_name,
                depth_path=depth_path,
                width=width,
                height=height,
                is_test=is_test,
            )
        )

    sys.stdout.write('\n')
    return cam_infos


# -----------------------------
# Scheme B: merge sparse/<sid>
# -----------------------------
def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8) -> SceneInfo:
    """
    Multi-sparse COLMAP loader (Scheme B) for structure:
      - images/<sid>/...
      - sparse/<sid>/...

    Split policy (YOU ASKED):
      - frame-level holdout: every 4 frames -> TEST
      - all cameras of that frame -> TEST
    """
    sparse_ids = _list_sparse_subdirs(path)
    if len(sparse_ids) == 0:
        raise FileNotFoundError(f"Could not find any sparse subdirs under: {os.path.join(path, 'sparse')}")

    reading_dir = "images" if images is None else images

    # ---- depth params: read once if depths enabled ----
    depths_params = None
    if depths != "":
        depth_params_file = os.path.join(path, "sparse", sparse_ids[0], "depth_params.json")
        if not os.path.exists(depth_params_file):
            print(f"Error: depth_params.json not found at '{depth_params_file}' but --depths is set.")
            sys.exit(1)

        with open(depth_params_file, "r") as f:
            depths_params = json.load(f)

        all_scales = np.array([depths_params[k]["scale"] for k in depths_params], dtype=np.float32)
        med_scale = np.median(all_scales[all_scales > 0]) if (all_scales > 0).sum() else 0.0
        for k in depths_params:
            depths_params[k]["med_scale"] = float(med_scale)

    # ---- Pass 1: read all views (is_test not decided yet) ----
    all_cam_infos: List[CameraInfo] = []
    for sid in sparse_ids:
        sdir = os.path.join(path, "sparse", sid)
        cam_extrinsics, cam_intrinsics = _read_colmap_pair(sdir)

        name_prefix = f"s{sid}_"

        cams_part = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            depths_params=depths_params,
            scene_root=path,
            reading_dir=reading_dir,
            depths_dir=depths,
            sid=sid,
            name_prefix=name_prefix,
            test_name_set=set(),  # decide after merge
        )
        all_cam_infos.extend(cams_part)

    # Stable order
    all_cam_infos = sorted(all_cam_infos, key=lambda x: x.image_name)

    # ---- Decide TEST set: frame-level holdout ----
    test_names_set = set()
    if eval:
        frame_hold = 4  # every 4th frame -> test (you asked)

        print("------------FRAME HOLDOUT (MERGED MULTI-SPARSE)-------------")
        print(f"[Split] frame_hold={frame_hold}: every {frame_hold}th frame -> TEST (all cameras)")

        # Collect all frame keys
        all_frames = sorted({_extract_frame_key_from_image_name(c.image_name) for c in all_cam_infos})

        # Choose test frames by index modulo
        test_frames = {fk for i, fk in enumerate(all_frames) if (i % frame_hold) == 0}

        # Mark all views of chosen frames as test
        for c in all_cam_infos:
            fk = _extract_frame_key_from_image_name(c.image_name)
            if fk in test_frames:
                test_names_set.add(c.image_name)

    # ---- Apply is_test flags ----
    cam_infos: List[CameraInfo] = []
    for c in all_cam_infos:
        cam_infos.append(
            CameraInfo(
                uid=c.uid,
                R=c.R,
                T=c.T,
                FovY=c.FovY,
                FovX=c.FovX,
                depth_params=c.depth_params,
                image_path=c.image_path,
                image_name=c.image_name,
                depth_path=c.depth_path,
                width=c.width,
                height=c.height,
                is_test=(c.image_name in test_names_set),
            )
        )

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # (Optional quick sanity print; safe to keep)
    try:
        from collections import Counter
        test_prefix = [c.image_name.split("_", 1)[0] for c in test_cam_infos]
        print(f"[SplitCheck] Train views: {len(train_cam_infos)}, Test views: {len(test_cam_infos)}")
        print("[SplitCheck] Test cam distribution:", Counter(test_prefix))
    except Exception:
        pass

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ---- points3D ----
    points_base = _find_points3d_base(path, sparse_ids)
    if points_base is None:
        points_base = os.path.join(path, "sparse", sparse_ids[0])

    ply_path = os.path.join(points_base, "points3D.ply")
    bin_path = os.path.join(points_base, "points3D.bin")
    txt_path = os.path.join(points_base, "points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting points3D.(bin/txt) to .ply (first time only).")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )


# -----------------------------
# Blender synthetic (unchanged)
# -----------------------------
def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(
                CameraInfo(
                    uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                    image_path=image_path, image_name=image_name,
                    width=image.size[0], height=image.size[1],
                    depth_path=depth_path, depth_params=None, is_test=is_test
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, (SH2RGB(shs) * 255))

    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=True
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
