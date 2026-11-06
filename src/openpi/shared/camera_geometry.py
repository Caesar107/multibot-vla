"""Camera geometry helpers for camera-conditioned policies (Plücker ray maps).

Follows "Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning"
(Jiang et al., ICRA 2026): every pixel is described by the Plücker coordinates of its viewing ray,
``[d, o x d]`` with ``o`` the camera center and ``d`` the unit ray direction, both in the world frame.

Conventions
-----------
* Cameras use the MuJoCo / OpenGL convention: in the camera frame +x is right, +y is up and the camera
  looks down **-z**. ``cam2world`` is the 4x4 matrix built from MuJoCo ``cam_xmat`` / ``cam_xpos``.
* The ray map is first computed for the *raw* MuJoCo render buffer, whose row 0 is the **bottom** of
  the image (OpenGL origin). openpi's LIBERO data and eval client rotate the raw render by 180 degrees
  (``img[::-1, ::-1]``); pass ``rotate_180=True`` to apply the same permutation to the ray map so pixels
  and rays stay aligned.
* ``K`` must be expressed at the same ``(height, width)`` as the requested ray map. Use
  :func:`scale_intrinsics` when the image was resized after rendering.

Everything here is numpy and runs inside data transforms (CPU), not inside the jitted model. The module is also
loaded by file path from the (python 3.8) LIBERO render environment in scripts/libero_campose, so keep it
dependency-free and 3.8-compatible.
"""

from __future__ import annotations

import numpy as np

PLUCKER_CHANNELS = 6

# Canonical LIBERO agentview at 256x256 (fovy=45), measured from MuJoCo after OffScreenRenderEnv reset.
# Used when training Plücker on the original single-camera `physical-intelligence/libero` dataset.
LIBERO_AGENTVIEW_FOVY_DEG = 45.0
LIBERO_AGENTVIEW_RENDER_HW = (256, 256)
LIBERO_AGENTVIEW_C2W = np.array(
    [
        [-7.2681843e-08, -6.2826645e-01, 7.7799827e-01, 6.5861315e-01],
        [1.0000000e00, 7.2681843e-08, 1.5211526e-07, 0.0],
        [-1.5211526e-07, 7.7799827e-01, 6.2826645e-01, 1.6103500e00],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def canonical_libero_camera(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat ``cam_K`` (9,) at ``(height, width)`` and ``cam_c2w`` (16,) for the stock LIBERO agentview."""
    k = intrinsics_from_fovy(LIBERO_AGENTVIEW_FOVY_DEG, *LIBERO_AGENTVIEW_RENDER_HW)
    k = scale_intrinsics(k, LIBERO_AGENTVIEW_RENDER_HW, (height, width))
    return k.reshape(-1).astype(np.float32), LIBERO_AGENTVIEW_C2W.reshape(-1).copy()


def intrinsics_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """Pinhole intrinsics from a MuJoCo vertical field of view (degrees), principal point at the image center."""
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float32)


def scale_intrinsics(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Rescale intrinsics after a plain (non-padded) resize from ``src_hw`` to ``dst_hw``."""
    K = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def cam2world_from_mujoco(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """4x4 cam2world from MuJoCo ``data.cam_xpos[i]`` (3,) and ``data.cam_xmat[i]`` (9,) or (3,3)."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(cam_xmat, dtype=np.float32).reshape(3, 3)
    c2w[:3, 3] = np.asarray(cam_xpos, dtype=np.float32).reshape(3)
    return c2w


def ray_directions(K: np.ndarray, cam2world: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit world-frame ray directions for every pixel of the raw (OpenGL, row 0 = bottom) render.

    Returns ``(origins, dirs)`` with shapes ``(3,)`` and ``(height, width, 3)``.
    """
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    c2w = np.asarray(cam2world, dtype=np.float32).reshape(4, 4)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Pixel centers. Row index grows upwards in the raw OpenGL buffer, so +row is +y in the camera frame.
    cols = np.arange(width, dtype=np.float32) + 0.5
    rows = np.arange(height, dtype=np.float32) + 0.5
    u, v = np.meshgrid(cols, rows, indexing="xy")  # (H, W)
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = -np.ones_like(x_cam)  # MuJoCo cameras look down -z
    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-9
    return c2w[:3, 3].copy(), dirs_world.astype(np.float32)


def plucker_raymap(
    K: np.ndarray,
    cam2world: np.ndarray,
    height: int,
    width: int,
    *,
    rotate_180: bool,
) -> np.ndarray:
    """Per-pixel Plücker coordinates ``[d, o x d]`` as a ``(height, width, 6)`` float32 array.

    Args:
        K: 3x3 intrinsics at ``(height, width)`` resolution.
        cam2world: 4x4 camera-to-world transform (MuJoCo convention, -z forward).
        height, width: output resolution.
        rotate_180: apply ``[::-1, ::-1]`` so the map matches images that were rotated by 180 degrees after
            rendering (openpi LIBERO convention). Use ``False`` for raw MuJoCo renders.
    """
    origin, dirs = ray_directions(K, cam2world, height, width)
    moment = np.cross(np.broadcast_to(origin, dirs.shape), dirs)
    raymap = np.concatenate([dirs, moment], axis=-1).astype(np.float32)
    if rotate_180:
        raymap = raymap[::-1, ::-1]
    return np.ascontiguousarray(raymap)


def plucker_from_flat(
    cam_K: np.ndarray,
    cam_c2w: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    rotate_180: bool,
) -> np.ndarray:
    """Convenience wrapper for dataset records: flat ``cam_K`` (9,) at ``src_hw`` and ``cam_c2w`` (16,)."""
    K = scale_intrinsics(np.asarray(cam_K, dtype=np.float32).reshape(3, 3), src_hw, dst_hw)
    return plucker_raymap(K, np.asarray(cam_c2w, dtype=np.float32).reshape(4, 4), *dst_hw, rotate_180=rotate_180)
