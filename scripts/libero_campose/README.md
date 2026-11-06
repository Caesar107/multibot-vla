# CamPose (Plücker camera conditioning) in openpi + LIBERO multi-camera data

Goal: test whether conditioning π0.5 on the camera's Plücker ray map (as in CamPoseOpensource,
"Do You Know Where Your Camera Is?") plus camera-pose randomization makes a LIBERO policy robust to the
LIBERO-Plus **Camera** perturbations.

## What was added to openpi

| Piece | Where |
| --- | --- |
| Plücker ray map from K / cam2world (MuJoCo/OpenGL convention, 180° rotation like the LIBERO images) | `src/openpi/shared/camera_geometry.py` (+ `_test.py`) |
| `Observation.plucker` (optional dict per image key), joint geometric augmentation with the RGB image | `src/openpi/models/model.py` |
| Zero-init parallel patch conv `plucker_embedding` (6→1152) added to SigLIP patch tokens | `src/openpi/models/siglip.py` |
| `Pi0Config.plucker_image_keys`, plumbing through `embed_prefix` / `fake_obs` / `inputs_spec` | `src/openpi/models/pi0_config.py`, `pi0.py` |
| `CheckpointWeightLoader.missing_regex` allows fresh `.*plucker.*` params | `src/openpi/training/weight_loaders.py` |
| `LiberoInputs` builds the ray map from `observation/cam_K` + `observation/cam_c2w` | `src/openpi/policies/libero_policy.py` |
| `LeRobotLiberoDataConfig.with_camera_params`, configs `pi05_libero_campose` / `pi05_libero_campose_canon` | `src/openpi/training/config.py` |
| LIBERO-Plus eval client that sends `cam_K` / `cam_c2w` (`--args.no-cam-cond` to disable) | `examples/libero_plus/` |

With the zero-init conv the model is numerically identical to stock π0.5 at step 0; only `base_0_rgb`
(agentview) is conditioned. `cam_K` is always expressed at the resolution of the image it accompanies
(256 in the dataset, 224 in the eval client after `resize_with_pad`).

Configs:

- `pi05_libero_campose` – view-randomized data **with** Plücker conditioning.
- `pi05_libero_multiview` – the same view-randomized data **without** Plücker (images only).
- `pi05_libero_campose_canon` – Plücker on the **original** `physical-intelligence/libero` data
  (fixed canonical camera; K/c2w injected, not stored in the dataset). Tests whether conditioning
  alone, without view-randomized data, transfers to unseen camera poses.
- `pi05_libero` – stock baseline (same original data, no camera conditioning).

## Data

| Stage | Path |
| --- | --- |
| Official LIBERO hdf5 (`yifengzhu-hf/LIBERO-datasets`, 4 suites × 10 tasks × 50 demos) | `/mnt/oss/zml/datasets/libero_raw/<suite>/*_demo.hdf5` |
| MuJoCo states / actions per task (npz) | `/mnt/workspace/zml/tmp/libero_campose/demo_states/<suite>/<task>.npz` |
| Rendered multi-view npz (scratch) | `/tmp/zml-libero-campose/render/<suite>/<task>/demo_<i>.npz` |
| LeRobot v2.1 dataset (`repo_id=libero_campose`) | `/mnt/oss/zml/datasets/pi/libero_campose` |

Each demo is rendered at 256² with **4 agentview cameras**: view 0 is the canonical LIBERO camera,
views 1–3 are sampled per demo using the LIBERO-Plus parameterization
(`horizon` U(−75°, 75°), `vertical` U(−10°, 25°), `scale` U(0.85, 2.0), end-point tilt U(−12°, 12°)).
Wrist image / state / actions are shared across views. OpenVLA's no-op filter is applied like the
stock `physical-intelligence/libero` dataset. Frame `t` is rendered from `states[t+1]`
(`--state-offset 1`), which reproduces the stored hdf5 observations (verified pixel-wise).

Dataset features: `image`, `wrist_image` (256²), `state` (8), `actions` (7), `cam_K` (9), `cam_c2w` (16),
`view_id` (1). One LeRobot episode per (demo, view) → ~8000 episodes.

## Two venvs

- **openpi venv** (py3.11, JAX, lerobot, h5py): `source /mnt/workspace/zml/code/multi-agent/openpi/env.sh`
  → use `python ...` (never `uv run` here, the venv is shared with `/mnt/workspace/zml/code/openpi`).
- **render / eval client venv** (py3.8, robosuite, LIBERO):
  `/mnt/workspace/zml/code/effectvla/openpi/examples/libero/.venv/bin/python`, with
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json`.
  Rendering uses the original LIBERO (`LIBERO_CONFIG_PATH=/mnt/workspace/zml/tmp/libero_campose/libero_config`);
  the eval client uses the LIBERO-Plus checkout from effectvla (`PYTHONPATH` set by `run_client.sh`).

## Reproduce the data

```bash
cd /mnt/workspace/zml/code/multi-agent/openpi
source env.sh

# 1. raw hdf5 -> OSS (hf-mirror)
python scripts/libero_campose/download_libero_raw.py

# 2. hdf5 -> states/actions npz (openpi venv has h5py)
python scripts/libero_campose/extract_demo_states.py

# 3. re-render 4 views per demo, 10 tasks in parallel (render venv, EGL)
bash scripts/libero_campose/run_render_all.sh 10
#    single task / pilot:
#    .../examples/libero/.venv/bin/python scripts/libero_campose/render_multicam.py --suite libero_spatial --task-index 0 --demos 0 1

# 4. npz -> LeRobot v2.1 (local scratch first, then move to OSS)
python scripts/libero_campose/convert_to_lerobot.py --render-root /tmp/zml-libero-campose/render \
    --out-root /tmp/zml-libero-campose/lerobot --repo-id libero_campose
mv /tmp/zml-libero-campose/lerobot/libero_campose /mnt/oss/zml/datasets/pi/libero_campose

# 5. norm stats (written to /mnt/oss/zml/models/openpi/assets/<config>/libero_campose for both configs).
#    Reads only the state/actions parquet columns (~30 s); the stock compute_norm_stats.py decodes every PNG (~3 h).
python scripts/libero_campose/fast_norm_stats.py --dataset-root /mnt/oss/zml/datasets/pi/libero_campose
```

Dataset as built: 8000 episodes, 1,351,676 frames, 40 tasks, 160 GB. Norm stats are within a few % of the
stock `pi05_libero` ones (same proprio/actions, 4× replicated across views).

## Train

Recipe = the outer fork's `pi05_libero` (π0.5 base weights, batch 256, cosine LR 5e-5, 30k steps,
EMA 0.999, `fsdp_devices=2`), run on DLC 8×80GB like the effectvla jobs. Checkpoints go to
`/mnt/oss/zml/checkpoints/openpi/<config>/<exp_name>/`, wandb stays on (key loaded by `env.sh`).

```bash
# DLC launch command (computes norm stats on first run, resumes if the exp dir has checkpoints)
bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh                      # pi05_libero_campose / campose_v1
CONFIG=pi05_libero_campose_canon EXP_NAME=campose_canon_v1 bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
CONFIG=pi05_libero_multiview EXP_NAME=multiview_v1 bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
```

Manual (any machine with the shared venv):

```bash
cd /mnt/workspace/zml/code/multi-agent/openpi && source env.sh
python scripts/compute_norm_stats.py --config-name pi05_libero_campose
python scripts/train.py pi05_libero_campose --exp_name campose_v1 --overwrite
```

Single A100-80GB (this DSW): a full fine-tune of π0.5 with EMA + AdamW does **not** fit (OOM even at
batch 4 with ~70 GB free). It runs with `--fsdp_devices 1 --ema_decay None --batch_size 4` (verified) if
you only need a local sanity run; for a real run use DLC.

Smoke test that was run (tiny dataset converted from the 2-demo pilot, 30 steps):

```bash
HF_LEROBOT_HOME=/tmp/zml-libero-campose/lerobot_test XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 \
python scripts/train.py pi05_libero_campose --exp_name smoke --data.repo_id libero_campose_test \
    --assets_base_dir /tmp/zml-libero-campose/smoke_assets --checkpoint_base_dir /tmp/zml-libero-campose/smoke_ckpt \
    --fsdp_devices 1 --ema_decay None --batch_size 4 --num_train_steps 30 --log_interval 1 --save_interval 30 \
    --num_workers 0 --overwrite --no-wandb-enabled
```

Result: `pi05_libero_campose` and a no-Plücker run give the identical step-0 loss (0.1140) on the
same batch — the zero-init Plücker conv leaves the stock model untouched — while `grad_norm` differs
(0.9456 vs 0.9354) because the new kernel already receives gradient. The step-29 checkpoint contains
`PaliGemma/img/plucker_embedding/kernel` (14,14,6,1152) with non-zero values.

## Evaluate on LIBERO-Plus (Camera category)

`run_full_eval.sh` starts the policy server from this repo, runs the client (effectvla's py3.8 venv +
LIBERO-Plus checkout, read-only), and aggregates. `CATEGORY=Camera` keeps only the camera-viewpoint
tasks (~376 per suite).

```bash
cd /mnt/workspace/zml/code/multi-agent/openpi

# Plücker model
CONFIG=pi05_libero_campose EXP_NAME=campose_v1 CATEGORY=Camera bash examples/libero_plus/run_full_eval.sh
# Plücker on original (fixed-camera) LIBERO
CONFIG=pi05_libero_campose_canon EXP_NAME=campose_canon_v1 CATEGORY=Camera bash examples/libero_plus/run_full_eval.sh
# stock pi05_libero (official checkpoint cached on OSS; camera keys are ignored by this config)
CONFIG=pi05_libero CKPT=/mnt/oss/zml/models/openpi/openpi-assets/checkpoints/pi05_libero EXP_NAME=stock \
  CATEGORY=Camera bash examples/libero_plus/run_full_eval.sh

# options: SUITE_ONLY=libero_spatial  SMOKE=1  NO_CAM_COND=1  CKPT=/path/to/step  PORT=8000 GPU=0
```

Results: `/mnt/oss/zml/logs/libero_campose/libero-plus-eval/<config>/<exp>/step<N>_Camera/`
(`*_summary.json` per suite, `leaderboard.json` after aggregation).

Manual two-process run:

```bash
# server (openpi venv)
source env.sh
python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_libero_campose \
    --policy.dir /mnt/oss/zml/checkpoints/openpi/pi05_libero_campose/campose_v1/29999
# client
bash examples/libero_plus/run_client.sh libero_spatial 8000 0 /path/to/out -- --args.category Camera
```

## Validation artefacts

`/mnt/oss/zml/logs/libero_campose/`:

- `pilot_contact_sheet.png` – 4 views × 2 frames of a pilot demo, eef projected with the stored K/c2w
  (red circle) and Plücker direction maps.
- `pilot_lerobot_vs_ours_frame0.png` – canonical re-render vs the stock LeRobot dataset frame
  (mean abs diff 1.6/255; state max diff 0.046, actions identical).
- `eval_client_cam_check.png` – LIBERO-Plus Camera tasks with eef projected using the client's K/c2w.
