"""Convert rendered multi-camera LIBERO demos (render_multicam.py npz) into a LeRobot v2.1 dataset.

One LeRobot episode per (demo, view). Mirrors ``physical-intelligence/libero`` (image mode, 256x256, fps 10,
``state`` 8-d, ``actions`` 7-d, OpenVLA no-op filter) and adds per-frame camera parameters:

    cam_K    (9,)  row-major 3x3 intrinsics at the stored image resolution (256x256)
    cam_c2w  (16,) row-major 4x4 camera-to-world (MuJoCo/OpenGL camera: -z forward, +y up)
    view_id  (1,)  0 = canonical LIBERO agentview, 1.. = randomized views

Run in the openpi venv (needs lerobot):

    source /mnt/workspace/zml/code/multi-agent/openpi/env.sh
    python scripts/libero_campose/convert_to_lerobot.py \
        --render-root /tmp/zml-libero-campose/render \
        --out-root /tmp/zml-libero-campose/lerobot --repo-id libero_campose

Write to local scratch first (OSS is slow for many small files), then move the finished directory to
``$HF_LEROBOT_HOME/<repo_id>`` (``/mnt/oss/zml/datasets/pi/libero_campose``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
IMAGE_SIZE = 256


def features(image_size: int) -> dict:
    return {
        "image": {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "cam_K": {"dtype": "float32", "shape": (9,), "names": ["cam_K"]},
        "cam_c2w": {"dtype": "float32", "shape": (16,), "names": ["cam_c2w"]},
        "view_id": {"dtype": "int64", "shape": (1,), "names": ["view_id"]},
    }


def list_demos(render_root: pathlib.Path, suites: tuple[str, ...]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for suite in suites:
        suite_dir = render_root / suite
        if not suite_dir.is_dir():
            print(f"[convert] WARNING: missing suite dir {suite_dir}")
            continue
        for task_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
            demos = sorted(task_dir.glob("demo_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
            files.extend(demos)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=pathlib.Path, default=pathlib.Path("/tmp/zml-libero-campose/render"))
    parser.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("/tmp/zml-libero-campose/lerobot"))
    parser.add_argument("--repo-id", default="libero_campose")
    parser.add_argument("--suites", nargs="*", default=list(SUITES))
    parser.add_argument("--views", type=int, nargs="*", default=None, help="subset of view ids (default: all)")
    parser.add_argument("--max-demos", type=int, default=None, help="debug: only convert the first N demo files")
    parser.add_argument("--no-noop-filter", action="store_true", help="keep OpenVLA no-op frames")
    parser.add_argument("--image-writer-threads", type=int, default=16)
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out_path = args.out_root / args.repo_id
    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    demo_files = list_demos(args.render_root, tuple(args.suites))
    if args.max_demos is not None:
        demo_files = demo_files[: args.max_demos]
    if not demo_files:
        raise SystemExit("no demo files found")
    print(f"[convert] {len(demo_files)} demo files from {args.render_root}", flush=True)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=out_path,
        robot_type="panda",
        fps=10,
        features=features(IMAGE_SIZE),
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    t0 = time.time()
    n_eps = n_frames = n_dropped = 0
    for n, f in enumerate(demo_files):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        language = meta["language"]
        keep = d["keep_mask"] if not args.no_noop_filter else np.ones(len(d["actions"]), bool)
        n_dropped += int((~keep).sum())
        views = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("agentview_"))
        if args.views is not None:
            views = [v for v in views if v in set(args.views)]
        wrist = d["wrist"]
        state = d["state"]
        actions = d["actions"]
        if wrist.shape[1] != IMAGE_SIZE:
            raise ValueError(f"{f}: expected {IMAGE_SIZE}px renders, got {wrist.shape}")
        for v in views:
            agent = d[f"agentview_{v}"]
            cam_k = d[f"cam_K_{v}"].astype(np.float32)
            cam_c2w = d[f"cam_c2w_{v}"].astype(np.float32)
            view_id = np.array([v], dtype=np.int64)
            for t in np.flatnonzero(keep):
                dataset.add_frame(
                    {
                        "image": agent[t],
                        "wrist_image": wrist[t],
                        "state": state[t],
                        "actions": actions[t],
                        "cam_K": cam_k,
                        "cam_c2w": cam_c2w,
                        "view_id": view_id,
                        "task": language,
                    }
                )
            dataset.save_episode()
            # LeRobot keeps every saved episode concatenated in memory (with embedded PNG bytes); we only need
            # the on-disk parquet + meta, so reset the in-memory table to keep RAM bounded.
            dataset.hf_dataset = dataset.create_hf_dataset()
            n_eps += 1
            n_frames += int(keep.sum())
        if (n + 1) % 10 == 0 or n + 1 == len(demo_files):
            el = time.time() - t0
            print(
                f"[convert] {n + 1}/{len(demo_files)} demos -> {n_eps} episodes, {n_frames} frames "
                f"(dropped {n_dropped} no-op frames/demo-views) elapsed={el:.0f}s "
                f"eta={el / (n + 1) * (len(demo_files) - n - 1):.0f}s",
                flush=True,
            )

    dataset.stop_image_writer()
    print(f"[convert] done: {n_eps} episodes, {n_frames} frames -> {out_path}")
    print(f"[convert] move to OSS with:  mv {out_path} /mnt/oss/zml/datasets/pi/{args.repo_id}")


if __name__ == "__main__":
    main()
