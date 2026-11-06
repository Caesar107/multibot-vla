"""Stage 0: dump MuJoCo states / actions / language of the official LIBERO hdf5 demos to small per-task npz files.

The LIBERO render environment (python 3.8) has no h5py, so the heavy hdf5 files are read here (openpi .venv) and
only what the renderer needs is written out:
    <out>/<suite>/<task>.npz  with  states_<i> (T, S) f64, actions_<i> (T, 7) f32, ref_agentview_<i> (T,128,128,3)
    (the stored 128x128 renders, used by the pilot to verify the state/observation alignment), plus
    language, bddl_file_name, num_demos.

Usage:
    python scripts/libero_campose/extract_demo_states.py --suites libero_spatial libero_object libero_goal libero_10
"""

import argparse
import json
import pathlib

import h5py
import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def extract_task(hdf5_path: pathlib.Path, out_path: pathlib.Path, *, keep_reference_images: bool) -> dict:
    arrays = {}
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        problem_info = json.loads(data.attrs["problem_info"])
        demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        for i, demo in enumerate(demos):
            g = data[demo]
            arrays[f"states_{i}"] = g["states"][()].astype(np.float64)
            arrays[f"actions_{i}"] = g["actions"][()].astype(np.float32)
            arrays[f"ee_states_{i}"] = g["obs/ee_states"][()].astype(np.float32)
            arrays[f"gripper_states_{i}"] = g["obs/gripper_states"][()].astype(np.float32)
            if keep_reference_images:
                arrays[f"ref_agentview_{i}"] = g["obs/agentview_rgb"][()]
        meta = {
            "language": problem_info["language_instruction"],
            "problem_name": problem_info["problem_name"],
            "bddl_file_name": str(data.attrs["bddl_file_name"]),
            "num_demos": len(demos),
            "source": str(hdf5_path),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=json.dumps(meta), **arrays)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=list(SUITES))
    parser.add_argument("--hdf5-root", type=pathlib.Path, default=pathlib.Path("/mnt/oss/zml/datasets/libero_raw"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/mnt/workspace/zml/tmp/libero_campose/demo_states"))
    parser.add_argument("--reference-images", action="store_true", help="also keep the stored 128x128 agentview renders")
    parser.add_argument("--tasks", nargs="*", default=None, help="optional task-name filter (substring)")
    args = parser.parse_args()

    for suite in args.suites:
        files = sorted((args.hdf5_root / suite).glob("*_demo.hdf5"))
        if args.tasks:
            files = [f for f in files if any(t in f.name for t in args.tasks)]
        for f in files:
            task = f.name[: -len("_demo.hdf5")]
            out = args.out / suite / f"{task}.npz"
            if out.exists():
                print(f"[extract] skip {suite}/{task} (exists)")
                continue
            meta = extract_task(f, out, keep_reference_images=args.reference_images)
            print(f"[extract] {suite}/{task}: {meta['num_demos']} demos -> {out}")


if __name__ == "__main__":
    main()
