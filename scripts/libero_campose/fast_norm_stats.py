"""Norm stats for libero_campose straight from the parquet columns (seconds instead of hours).

`scripts/compute_norm_stats.py` decodes every PNG only to read `state` / `actions`. For the LIBERO configs the
data transforms leave both untouched (no padding, no delta), so the result is the same `RunningStats` over the raw
`state` column and over horizon-H action chunks (LeRobot clamps the chunk to the last frame of the episode).

    source /mnt/workspace/zml/code/multi-agent/openpi/env.sh
    python scripts/libero_campose/fast_norm_stats.py --dataset-root /mnt/oss/zml/datasets/pi/libero_campose
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("/mnt/oss/zml/datasets/pi/libero_campose"))
    parser.add_argument("--repo-id", default="libero_campose")
    parser.add_argument("--configs", nargs="*", default=["pi05_libero_campose", "pi05_libero_campose_canon"])
    args = parser.parse_args()

    horizons = {_config.get_config(c).model.action_horizon for c in args.configs}
    if len(horizons) != 1:
        raise SystemExit(f"configs have different action horizons: {horizons}")
    horizon = horizons.pop()

    files = sorted(args.dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.dataset_root}")
    print(f"{len(files)} episodes, action_horizon={horizon}")
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    t0 = time.time()
    for n, f in enumerate(files):
        table = pq.read_table(f, columns=["state", "actions"])
        state = np.stack(table.column("state").to_numpy(zero_copy_only=False)).astype(np.float32)
        actions = np.stack(table.column("actions").to_numpy(zero_copy_only=False)).astype(np.float32)
        num_frames = len(actions)
        idx = np.minimum(np.arange(num_frames)[:, None] + np.arange(horizon)[None, :], num_frames - 1)
        stats["state"].update(state)
        stats["actions"].update(actions[idx])
        if (n + 1) % 2000 == 0:
            print(f"{n + 1}/{len(files)} {time.time() - t0:.0f}s")

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    for k, v in norm_stats.items():
        print(k, "mean", np.round(v.mean, 3), "std", np.round(v.std, 3))
    for cfg in args.configs:
        out = pathlib.Path(_config.get_config(cfg).assets_dirs) / args.repo_id
        normalize.save(out, norm_stats)
        print("wrote", out / "norm_stats.json")


if __name__ == "__main__":
    main()
