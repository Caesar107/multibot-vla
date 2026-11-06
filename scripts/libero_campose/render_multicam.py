"""Stage 1: re-render official LIBERO demos from several agentview camera poses (CamPose-style view randomization).

Runs inside the LIBERO render environment (python 3.8, robosuite + original LIBERO, MuJoCo EGL), e.g.

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json \
    LIBERO_CONFIG_PATH=/mnt/workspace/zml/tmp/libero_campose/libero_config \
    PYTHONPATH=/mnt/workspace/zml/code/openpi/third_party/libero \
    /mnt/workspace/zml/code/effectvla/openpi/examples/libero/.venv/bin/python \
        scripts/libero_campose/render_multicam.py --suite libero_spatial --task-index 0

For every demo and every view the full MuJoCo state sequence of the demo is replayed with
`sim.set_state_from_flattened`, the agentview camera is moved to the sampled pose, and the agentview + wrist images
are rendered at 256x256. Actions and proprioceptive state are *unchanged* by the camera. View 0 is the canonical
(unperturbed) pose; views 1..V-1 are random perturbations parameterized exactly like LIBERO-Plus "Camera
Viewpoints" (orbit about world z, pitch about a y-axis through (0,0,0.8), dolly from (0,0,0.8), look-at tilt),
sampled from continuous ranges that are slightly wider than the LIBERO-Plus test catalogue.

Output: <out>/<suite>/<task>/demo_<i>.npz with, per view v:
    agentview_<v> (T,256,256,3) uint8   rotated 180 deg (openpi/OpenVLA LIBERO convention)
    cam_K_<v> (9,) f32 at 256x256, cam_c2w_<v> (16,) f32 (MuJoCo camera-to-world, -z forward)
    cam_params_<v> (5,) f32 = [horizon_deg, vertical_deg, scale, end_rot_deg, end_vert_deg]
and shared: wrist (T,256,256,3) uint8 (rotated 180), state (T,8) f32 = [eef_pos, eef_axis_angle, gripper_qpos],
actions (T,7) f32, keep_mask (T,) bool (OpenVLA no-op filter), meta json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import time
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = pathlib.Path(__file__).resolve().parent
_CG_PATH = _HERE.parents[1] / "src" / "openpi" / "shared" / "camera_geometry.py"
_spec = importlib.util.spec_from_file_location("camera_geometry", _CG_PATH)
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)  # type: ignore[union-attr]

IMAGE_SIZE = 256
CAMERA_NAME = "agentview"
PIVOT = np.array([0.0, 0.0, 0.8])

# Training distribution of camera perturbations (degrees / scale). LIBERO-Plus test set: horizon +-1..75,
# vertical {0, 15}, scale 1.15..2.0, end tilt +-2..10. Slightly wider so test poses are unseen but in-distribution.
DEFAULT_RANGES = {
    "horizon": (-75.0, 75.0),
    "vertical": (-10.0, 25.0),
    "scale": (0.85, 2.0),
    "end_rot": (-12.0, 12.0),
    "end_vert": (-12.0, 12.0),
}


# --- LIBERO-Plus camera helpers (same math/pivots as LIBERO-plus/libero/libero/envs/problems/*.py) -------------


def _quat_wxyz_to_rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _rot_to_quat_wxyz(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rotate_around_y(quat_wxyz, pos, degrees):
    rot = Rotation.from_rotvec(np.radians(-degrees) * np.array([0.0, 1.0, 0.0]))
    new_quat = _rot_to_quat_wxyz(rot * _quat_wxyz_to_rot(quat_wxyz))
    new_pos = None if pos is None else rot.apply(np.asarray(pos) - PIVOT) + PIVOT
    return new_quat, new_pos


def rotate_around_z(quat_wxyz, pos, degrees):
    rot = Rotation.from_euler("z", degrees, degrees=True)
    new_quat = _rot_to_quat_wxyz(rot * _quat_wxyz_to_rot(quat_wxyz))
    new_pos = None if pos is None else rot.apply(np.asarray(pos))
    return new_quat, new_pos


def scale_from_pivot(pos, scale):
    return PIVOT + (np.asarray(pos) - PIVOT) * scale


def perturb_camera(base_pos, base_quat_wxyz, horizon, vertical, scale, end_rot, end_vert):
    """Apply LIBERO-Plus style perturbations in LIBERO-Plus order. Returns (pos, quat_wxyz)."""
    quat, pos = np.asarray(base_quat_wxyz, dtype=np.float64), np.asarray(base_pos, dtype=np.float64)
    if vertical != 0.0:
        quat, pos = rotate_around_y(quat, pos, vertical)
    quat, pos = rotate_around_z(quat, pos, horizon)
    if scale != 1.0:
        pos = scale_from_pivot(pos, scale)
    if end_rot != 0.0:
        quat, _ = rotate_around_z(quat, None, end_rot)
    if end_vert != 0.0:
        quat, _ = rotate_around_y(quat, None, end_vert)
    return pos, quat


def sample_params(rng: np.random.Generator, ranges=DEFAULT_RANGES):
    return np.array(
        [
            rng.uniform(*ranges["horizon"]),
            rng.uniform(*ranges["vertical"]),
            rng.uniform(*ranges["scale"]),
            rng.uniform(*ranges["end_rot"]),
            rng.uniform(*ranges["end_vert"]),
        ],
        dtype=np.float64,
    )


CANONICAL_PARAMS = np.array([0.0, 0.0, 1.0, 0.0, 0.0])


# --- rendering --------------------------------------------------------------------------------------------


def noop_keep_mask(actions: np.ndarray) -> np.ndarray:
    """OpenVLA LIBERO no-op filter: drop frames whose arm action is all-zero and whose gripper command is unchanged."""
    keep = np.ones(len(actions), dtype=bool)
    for t in range(len(actions)):
        if np.all(actions[t, :6] == 0.0):
            prev_grip = actions[t - 1, 6] if t > 0 else None
            if prev_grip is not None and actions[t, 6] == prev_grip:
                keep[t] = False
    return keep


def quat2axisangle(quat_xyzw):
    import robosuite.utils.transform_utils as T

    return T.quat2axisangle(np.asarray(quat_xyzw, dtype=np.float64))


class DemoRenderer:
    def __init__(self, bddl_file: str, image_size: int = IMAGE_SIZE):
        from libero.libero.envs import OffScreenRenderEnv

        self.env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=image_size, camera_widths=image_size)
        self.env.seed(0)
        self.env.reset()
        self.image_size = image_size
        self.sim = self.env.sim
        self.cam_id = self.sim.model.camera_name2id(CAMERA_NAME)
        self.base_pos = np.array(self.sim.model.cam_pos[self.cam_id], dtype=np.float64)
        self.base_quat = np.array(self.sim.model.cam_quat[self.cam_id], dtype=np.float64)  # wxyz
        self.fovy = float(self.sim.model.cam_fovy[self.cam_id])

    def set_camera(self, pos, quat_wxyz):
        self.sim.model.cam_pos[self.cam_id] = pos
        self.sim.model.cam_quat[self.cam_id] = quat_wxyz
        self.sim.forward()

    def camera_geometry(self):
        K = cg.intrinsics_from_fovy(self.fovy, self.image_size, self.image_size)
        c2w = cg.cam2world_from_mujoco(self.sim.data.cam_xpos[self.cam_id], self.sim.data.cam_xmat[self.cam_id])
        return K.reshape(-1).astype(np.float32), c2w.reshape(-1).astype(np.float32)

    def set_state(self, flat_state):
        self.sim.set_state_from_flattened(flat_state)
        self.sim.forward()

    def render_agentview(self):
        img = self.sim.render(camera_name=CAMERA_NAME, height=self.image_size, width=self.image_size, depth=False)
        return np.ascontiguousarray(img[::-1, ::-1])

    def observe(self):
        """Full observation (wrist image + proprio) at the current state. Also renders agentview."""
        self.env.env._update_observables(force=True)
        obs = self.env.env._get_observations()
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        state = np.concatenate(
            [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
        ).astype(np.float32)
        return agent, wrist, state

    def close(self):
        self.env.close()


def render_demo(r: DemoRenderer, states, actions, *, num_views, rng, state_offset, ranges):
    T_ = len(actions)
    # Frame t is rendered at states[min(t + state_offset, T-1)] (see --state-offset).
    idx = np.minimum(np.arange(T_) + state_offset, len(states) - 1)
    out = {}
    params_all = [CANONICAL_PARAMS] + [sample_params(rng, ranges) for _ in range(num_views - 1)]
    wrist = np.zeros((T_, r.image_size, r.image_size, 3), np.uint8)
    state = np.zeros((T_, 8), np.float32)
    for v, params in enumerate(params_all):
        pos, quat = perturb_camera(r.base_pos, r.base_quat, *params)
        r.set_camera(pos, quat)
        agent = np.zeros((T_, r.image_size, r.image_size, 3), np.uint8)
        for t in range(T_):
            r.set_state(states[idx[t]])
            if v == 0:
                agent[t], wrist[t], state[t] = r.observe()
            else:
                agent[t] = r.render_agentview()
        K, c2w = r.camera_geometry()
        out[f"agentview_{v}"] = agent
        out[f"cam_K_{v}"] = K
        out[f"cam_c2w_{v}"] = c2w
        out[f"cam_params_{v}"] = params.astype(np.float32)
    # restore canonical camera for the next demo
    r.set_camera(r.base_pos, r.base_quat)
    out["wrist"] = wrist
    out["state"] = state
    out["actions"] = actions.astype(np.float32)
    out["keep_mask"] = noop_keep_mask(actions)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-index", type=int, default=None, help="index into sorted task list of the suite")
    parser.add_argument("--task", default=None, help="task name (file stem) instead of --task-index")
    parser.add_argument("--states-dir", type=pathlib.Path, default=pathlib.Path("/mnt/workspace/zml/tmp/libero_campose/demo_states"))
    parser.add_argument("--libero-root", type=pathlib.Path, default=pathlib.Path("/mnt/workspace/zml/code/openpi/third_party/libero"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/zml-libero-campose/render"))
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--demos", type=int, nargs="*", default=None, help="subset of demo indices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-offset",
        type=int,
        default=1,
        help="render frame t at states[t + offset]. Official LIBERO hdf5 stores obs[t] *after* actions[t], i.e. at "
        "states[t+1]; offset=1 reproduces the official (and OpenVLA/openpi) pairing.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_files = sorted((args.states_dir / args.suite).glob("*.npz"))
    if args.task is not None:
        task_files = [f for f in task_files if f.stem == args.task]
    elif args.task_index is not None:
        task_files = [task_files[args.task_index]]
    if len(task_files) != 1:
        raise SystemExit(f"expected exactly one task, got {[f.stem for f in task_files]}")
    task_file = task_files[0]
    task = task_file.stem
    data = np.load(task_file, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    bddl = args.libero_root / meta["bddl_file_name"]
    if not bddl.exists():
        raise FileNotFoundError(bddl)

    out_dir = args.out / args.suite / task
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_ids = list(range(meta["num_demos"])) if args.demos is None else list(args.demos)
    todo = [i for i in demo_ids if args.overwrite or not (out_dir / f"demo_{i}.npz").exists()]
    print(f"[render] {args.suite}/{task}: {len(todo)}/{len(demo_ids)} demos to render, views={args.num_views}", flush=True)
    if not todo:
        return

    r = DemoRenderer(str(bddl), image_size=args.image_size)
    print(
        f"[render] canonical {CAMERA_NAME}: pos={r.base_pos.round(4).tolist()} quat={r.base_quat.round(4).tolist()} "
        f"fovy={r.fovy} state_dim={r.sim.get_state().flatten().shape[0]} demo_state_dim={data['states_0'].shape[1]}",
        flush=True,
    )
    if r.sim.get_state().flatten().shape[0] != data["states_0"].shape[1]:
        raise RuntimeError("MuJoCo state size mismatch between env and demo; wrong bddl / LIBERO version?")

    t0 = time.time()
    for n, i in enumerate(todo):
        # Reproducible per (suite, task, demo) camera poses, independent of which demos are rendered in this run.
        rng = np.random.default_rng([args.seed, zlib.crc32(args.suite.encode()), zlib.crc32(task.encode()), i])
        out = render_demo(
            r,
            data[f"states_{i}"],
            data[f"actions_{i}"],
            num_views=args.num_views,
            rng=rng,
            state_offset=args.state_offset,
            ranges=DEFAULT_RANGES,
        )
        demo_meta = dict(meta, demo=i, suite=args.suite, task=task, state_offset=args.state_offset, seed=args.seed,
                         base_pos=r.base_pos.tolist(), base_quat_wxyz=r.base_quat.tolist(), fovy=r.fovy)
        tmp = out_dir / f"demo_{i}.npz.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(demo_meta), **out)
        os.replace(tmp, out_dir / f"demo_{i}.npz")
        el = time.time() - t0
        print(f"[render] demo {i} ({n + 1}/{len(todo)}) T={len(out['actions'])} keep={int(out['keep_mask'].sum())} "
              f"elapsed={el:.0f}s eta={(el / (n + 1)) * (len(todo) - n - 1):.0f}s", flush=True)
    r.close()


if __name__ == "__main__":
    main()
