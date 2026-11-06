import os

import numpy as np
import pytest

from openpi.shared import camera_geometry as cg


def _lookat_c2w(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo-convention cam2world (camera looks down -z, +y up)."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def test_center_pixel_is_optical_axis():
    H = W = 64
    K = cg.intrinsics_from_fovy(45.0, H, W)
    c2w = _lookat_c2w(eye=(1.0, 0.5, 1.5), target=(0.0, 0.0, 0.8))
    origin, dirs = cg.ray_directions(K, c2w, H, W)
    # Average of the 4 central pixels is the optical axis = -z of the camera frame.
    center = dirs[H // 2 - 1 : H // 2 + 1, W // 2 - 1 : W // 2 + 1].reshape(-1, 3).mean(0)
    center /= np.linalg.norm(center)
    np.testing.assert_allclose(center, -c2w[:3, 2], atol=1e-6)
    np.testing.assert_allclose(origin, c2w[:3, 3])
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-5)


def test_raw_buffer_orientation():
    """In the raw OpenGL buffer row 0 is the bottom: rays from row 0 point *below* rays from the last row."""
    H = W = 32
    K = cg.intrinsics_from_fovy(60.0, H, W)
    c2w = _lookat_c2w(eye=(2.0, 0.0, 1.0), target=(0.0, 0.0, 1.0))  # level camera looking along -x
    _, dirs = cg.ray_directions(K, c2w, H, W)
    assert dirs[0, W // 2, 2] < 0 < dirs[-1, W // 2, 2]
    # Camera +x is right = forward x up = (-x) x (+z) = +y: image-right is world +y.
    assert dirs[H // 2, -1, 1] > 0 > dirs[H // 2, 0, 1]


def test_plucker_rotate_180_matches_image_rotation():
    H, W = 24, 32
    K = cg.intrinsics_from_fovy(45.0, H, W)
    c2w = _lookat_c2w(eye=(1.0, 0.2, 1.4), target=(0.0, 0.0, 0.8))
    raw = cg.plucker_raymap(K, c2w, H, W, rotate_180=False)
    rot = cg.plucker_raymap(K, c2w, H, W, rotate_180=True)
    np.testing.assert_array_equal(rot, raw[::-1, ::-1])
    assert rot.shape == (H, W, 6)
    # Plücker moment is perpendicular to the direction.
    assert np.abs((rot[..., :3] * rot[..., 3:]).sum(-1)).max() < 1e-5


def test_scale_intrinsics_keeps_rays():
    K256 = cg.intrinsics_from_fovy(45.0, 256, 256)
    K224 = cg.scale_intrinsics(K256, (256, 256), (224, 224))
    np.testing.assert_allclose(K224, cg.intrinsics_from_fovy(45.0, 224, 224), rtol=1e-6)
    c2w = _lookat_c2w(eye=(0.66, 0.0, 1.61), target=(0.0, 0.0, 0.8))
    flat = cg.plucker_from_flat(K256.reshape(-1), c2w.reshape(-1), src_hw=(256, 256), dst_hw=(224, 224), rotate_180=True)
    direct = cg.plucker_raymap(K224, c2w, 224, 224, rotate_180=True)
    np.testing.assert_allclose(flat, direct, atol=1e-6)


_SPHERE_XML = """
<mujoco>
  <visual><global offwidth="128" offheight="96"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="sphere" size="0.04" pos="{x} {y} {z}" rgba="1 0 0 1"/>
    <camera name="cam" pos="{cx} {cy} {cz}" quat="{qw} {qx} {qy} {qz}" fovy="{fovy}"/>
  </worldbody>
</mujoco>
"""


def _quat_wxyz_from_mat(R):
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return q[3], q[0], q[1], q[2]


@pytest.mark.skipif(os.environ.get("MUJOCO_GL", "") == "", reason="needs MUJOCO_GL=egl for offscreen rendering")
def test_against_mujoco_render():
    """Project a red sphere with our rays and compare against where MuJoCo actually draws it (raw buffer)."""
    import mujoco

    H, W, fovy = 96, 128, 50.0
    eye, target = (1.2, 0.3, 1.4), (0.0, 0.0, 0.8)
    c2w = _lookat_c2w(eye, target)
    sphere = np.array([0.15, -0.1, 0.95])
    qw, qx, qy, qz = _quat_wxyz_from_mat(c2w[:3, :3])
    xml = _SPHERE_XML.format(
        x=sphere[0], y=sphere[1], z=sphere[2], cx=eye[0], cy=eye[1], cz=eye[2], qw=qw, qx=qx, qy=qy, qz=qz, fovy=fovy
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera="cam")
    renderer.render()
    # Re-read the framebuffer ourselves: this is the raw OpenGL (row 0 = bottom) buffer robosuite returns.
    raw = np.empty((H, W, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(raw, None, renderer._rect, renderer._mjr_context)
    red = (raw[..., 0] > 150) & (raw[..., 1] < 80) & (raw[..., 2] < 80)
    assert red.sum() > 10, "sphere not visible"
    rows, cols = np.nonzero(red)
    centroid = np.array([rows.mean(), cols.mean()])

    # Pixel whose ray is most aligned with the sphere direction.
    K = cg.intrinsics_from_fovy(fovy, H, W)
    c2w_mj = cg.cam2world_from_mujoco(data.cam_xpos[0], data.cam_xmat[0])
    np.testing.assert_allclose(c2w_mj, c2w, atol=1e-5)
    origin, dirs = cg.ray_directions(K, c2w_mj, H, W)
    to_sphere = sphere - origin
    to_sphere /= np.linalg.norm(to_sphere)
    best = np.unravel_index(np.argmax(dirs @ to_sphere), dirs.shape[:2])
    assert np.abs(np.array(best) - centroid).max() < 1.5, (best, centroid)
