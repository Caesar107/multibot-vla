"""Download the official LIBERO demonstration hdf5 files (with MuJoCo states) from HF.

The raw hdf5 files are needed to re-render demos from new camera poses. They are stored on OSS
(large data), never on NAS.

Usage (any python with huggingface_hub):
    HF_ENDPOINT=https://hf-mirror.com python scripts/libero_campose/download_libero_raw.py \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --output-dir /mnt/oss/zml/datasets/libero_raw
"""

import argparse
import os
import pathlib

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/mnt/oss/zml/datasets/libero_raw"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download

    for suite in args.suites:
        if suite not in ALL_SUITES:
            raise ValueError(f"unknown suite {suite}; choose from {ALL_SUITES}")

    api = HfApi()
    for suite in args.suites:
        files = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset") if f.startswith(f"{suite}/")]
        print(f"[download] {suite}: {len(files)} files -> {args.output_dir / suite}", flush=True)
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{suite}/*"],
            local_dir=str(args.output_dir),
            max_workers=args.workers,
        )
        have = sorted((args.output_dir / suite).glob("*.hdf5"))
        print(f"[download] {suite}: {len(have)} hdf5 on disk", flush=True)


if __name__ == "__main__":
    main()
