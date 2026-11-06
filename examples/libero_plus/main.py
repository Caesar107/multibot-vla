"""LIBERO-Plus eval client for an openpi policy server, with camera parameters (CamPose / Plücker).

Observation pipeline matches `examples/libero/main.py` (and sylvestf/openpi_liberoplus `run.py`):
render 256, rotate 180, resize 224, replan every 5 steps, seed 7. Gripper is left as the
policy outputs it.

In addition to the stock keys, every inference request carries the agentview camera geometry read
from the MuJoCo sim (`observation/cam_K` 3x3 flattened at the *resized* resolution,
`observation/cam_c2w` 4x4 flattened, MuJoCo/OpenGL camera convention). `LiberoInputs` turns these
into a Plücker ray map for configs with `plucker_image_keys`; configs without it ignore the keys.
Pass `--args.no-cam-cond` to omit them (e.g. to evaluate a stock policy against a strict server).

Task protocol matches the LIBERO-Plus paper (sylvestf/LIBERO-plus): each perturbed task is
already its own entry, so the default is 1 trial per task. Suites are about
spatial 2402 / object 2518 / goal 2591 / libero_10 2519 (10,030 total).

Category labels come from `libero/libero/benchmark/task_classification.json`.
The published leaderboard averages the four suite rates; category columns average
each suite's category rate. `--args.aggregate-only` prints that table from summary jsons.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.util
import json
import logging
import math
import pathlib

import imageio
import libero.libero
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
AGENTVIEW_CAMERA = "agentview"

# The client runs in the py3.8 LIBERO venv without openpi installed; load the (numpy-only) geometry helpers
# straight from this repo so training and eval share one definition of K / cam2world.
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CG_PATH = _OPENPI_ROOT / "src" / "openpi" / "shared" / "camera_geometry.py"
_spec = importlib.util.spec_from_file_location("openpi_camera_geometry", _CG_PATH)
camera_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(camera_geometry)

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Paper leaderboard column order. Raw names are the strings in task_classification.json.
CATEGORY_SHORT = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
LEADERBOARD_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LEADERBOARD_COLUMNS = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    # Short leaderboard name, e.g. Robot. Empty runs every task in the index range.
    category: str = ""

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    # LIBERO-Plus bakes each perturbation into its own task. The paper uses 1 trial.
    num_trials_per_task: int = 1
    seed: int = 7

    # Inclusive start, exclusive end. -1 end means the rest of the suite.
    start_idx: int = 0
    end_idx: int = -1

    output_dir: str = "data/libero_plus"
    save_videos: bool = False
    task_classification_path: str = ""

    # Send observation/cam_K + observation/cam_c2w read from the sim (needed by pi05_libero_campose).
    # `--args.no-cam-cond` disables it.
    cam_cond: bool = True

    # Skip rollouts and merge `{output_dir}/*_summary.json` into the paper's table.
    aggregate_only: bool = False


def eval_libero(args: Args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(output_dir)
        return

    np.random.seed(args.seed)
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Options: {list(MAX_STEPS)}")

    classification = _load_classification(args.task_classification_path, args.task_suite_name)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    start_idx = max(0, args.start_idx)
    end_idx = n_tasks if args.end_idx < 0 else min(args.end_idx, n_tasks)
    if start_idx >= end_idx:
        raise ValueError(f"Empty task range [{start_idx}, {end_idx}) of {n_tasks}")

    task_ids = list(range(start_idx, end_idx))
    if args.category:
        wanted = args.category.strip()
        selected = []
        for task_id in task_ids:
            meta = classification.get(task_suite.get_task(task_id).name)
            if meta is None:
                continue
            raw = meta["category"]
            short = CATEGORY_SHORT.get(raw, raw)
            if wanted in (short, raw):
                selected.append(task_id)
        if not selected:
            raise ValueError(f"No tasks with category {wanted!r} in [{start_idx}, {end_idx})")
        logging.info("Category filter %s kept %d/%d tasks", wanted, len(selected), len(task_ids))
        task_ids = selected

    max_steps = MAX_STEPS[args.task_suite_name]
    logging.info(
        "Suite %s tasks [%d, %d) of %d, selected=%d, trials/task=%d, max_steps=%d, replan=%d, category=%s",
        args.task_suite_name,
        start_idx,
        end_idx,
        n_tasks,
        len(task_ids),
        args.num_trials_per_task,
        max_steps,
        args.replan_steps,
        args.category or "all",
    )

    video_dir = output_dir / "videos" / args.task_suite_name
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    cat_tag = f"_{args.category}" if args.category else ""
    stem = f"{args.task_suite_name}{cat_tag}_{start_idx}_to_{end_idx}"
    jsonl_path = output_dir / f"{stem}_tasks.jsonl"
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    per_category: dict[str, dict[str, int]] = {}
    per_difficulty: dict[str, dict[str, int]] = {}

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for task_id in tqdm.tqdm(task_ids, desc=args.task_suite_name):
            task = task_suite.get_task(task_id)
            meta = classification.get(task.name)
            if meta is None:
                logging.warning("Task %s (id %d) is missing from task_classification.json", task.name, task_id)
                category, short, difficulty = "Unknown", "Unknown", -1
            else:
                category = meta["category"]
                short = CATEGORY_SHORT.get(category, category)
                raw_difficulty = meta.get("difficulty_level")
                # Some LIBERO-Plus light variants leave this field null.
                difficulty = int(raw_difficulty) if raw_difficulty is not None else -1

            initial_states = task_suite.get_task_init_states(task_id)
            n_states = int(initial_states.shape[0])
            n_episodes = min(args.num_trials_per_task, n_states)
            if n_episodes < args.num_trials_per_task:
                logging.info("Task %d has %d init states; running %d", task_id, n_states, n_episodes)

            successes = _rollout_task(
                args,
                client,
                task,
                initial_states,
                n_episodes,
                max_steps,
                video_dir,
                f"{task_id:04d}_{short}",
            )
            n_succ = int(sum(successes))
            total_episodes += len(successes)
            total_successes += n_succ
            _bump(per_category, short, len(successes), n_succ)
            _bump(per_difficulty, str(difficulty), len(successes), n_succ)

            record = {
                "task_id": task_id,
                "name": task.name,
                "language": task.language,
                "category": short,
                "category_raw": category,
                "difficulty_level": difficulty,
                "successes": n_succ,
                "episodes": len(successes),
            }
            jsonl.write(json.dumps(record, ensure_ascii=True) + "\n")
            jsonl.flush()
            logging.info(
                "task %d [%s] %d/%d  running %d/%d (%.1f%%)",
                task_id,
                short,
                n_succ,
                len(successes),
                total_successes,
                total_episodes,
                100.0 * total_successes / total_episodes,
            )

    summary = {
        "task_suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "category_filter": args.category,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "per_category": per_category,
        "per_difficulty": per_difficulty,
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Suite success rate %.4f (%d/%d)", summary["success_rate"], total_successes, total_episodes)
    _log_buckets("category", per_category)
    logging.info("Wrote %s", summary_path)


def _rollout_task(args, client, task, initial_states, n_episodes, max_steps, video_dir, video_tag):
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    successes = []
    try:
        for episode_idx in range(n_episodes):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            done = False
            replay_images = []
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        if args.cam_cond:
                            cam_k, cam_c2w = _camera_params(env, LIBERO_ENV_RESOLUTION, args.resize_size)
                            element["observation/cam_K"] = cam_k
                            element["observation/cam_c2w"] = cam_c2w
                        action_chunk = client.infer(element)["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Policy predicts {len(action_chunk)} steps, replan_steps={args.replan_steps}"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                    t += 1
                except Exception as exc:
                    logging.error("Caught exception on task %s episode %d: %s", task.name, episode_idx, exc)
                    done = False
                    break

            successes.append(bool(done))
            if args.save_videos and replay_images:
                suffix = "success" if done else "failure"
                imageio.mimwrite(
                    video_dir / f"rollout_{video_tag}_ep{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
    finally:
        env.close()
    return successes


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _camera_params(env, render_size: int, resize_size: int):
    """Agentview intrinsics (flattened 3x3, at `resize_size`) and cam2world (flattened 4x4) from the live sim.

    Matches `render_multicam.py`: K from `cam_fovy` at the render resolution, then rescaled to the resolution
    of the image actually sent (resize_with_pad 256->224 is a pure scale, no padding). c2w from
    `data.cam_xpos/cam_xmat`, so LIBERO-Plus camera perturbations are picked up automatically.
    """
    sim = env.sim
    cam_id = sim.model.camera_name2id(AGENTVIEW_CAMERA)
    k = camera_geometry.intrinsics_from_fovy(float(sim.model.cam_fovy[cam_id]), render_size, render_size)
    k = camera_geometry.scale_intrinsics(k, (render_size, render_size), (resize_size, resize_size))
    c2w = camera_geometry.cam2world_from_mujoco(sim.data.cam_xpos[cam_id], sim.data.cam_xmat[cam_id])
    return k.reshape(-1).astype(np.float32), c2w.reshape(-1).astype(np.float32)


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils.quat2axisangle, same as examples/libero/main.py."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _load_classification(path: str, suite: str) -> dict[str, dict]:
    if path:
        classification_path = pathlib.Path(path)
    else:
        classification_path = pathlib.Path(libero.libero.__file__).parent / "benchmark" / "task_classification.json"
    with classification_path.open(encoding="utf-8") as f:
        mapping = json.load(f)
    if suite not in mapping:
        raise KeyError(f"Suite {suite} not in {classification_path}. Keys: {list(mapping)}")
    by_name = {}
    for item in mapping[suite]:
        by_name[item["name"]] = item
    logging.info("Loaded %d classification entries from %s", len(by_name), classification_path)
    return by_name


def _bump(buckets: dict[str, dict[str, int]], key: str, episodes: int, successes: int) -> None:
    slot = buckets.setdefault(key, {"total": 0, "success": 0})
    slot["total"] += episodes
    slot["success"] += successes


def _log_buckets(title: str, buckets: dict[str, dict[str, int]]) -> None:
    for key in list(LEADERBOARD_COLUMNS) + [k for k in buckets if k not in LEADERBOARD_COLUMNS]:
        if key not in buckets:
            continue
        slot = buckets[key]
        rate = slot["success"] / slot["total"] if slot["total"] else 0.0
        logging.info("%s %-12s %6.2f%%  (%d/%d)", title, key, 100.0 * rate, slot["success"], slot["total"])


def _aggregate(output_dir: pathlib.Path) -> None:
    """Paper table: Total is the mean of suite micro rates; each column is the mean of per-suite rates."""
    merged: dict[str, dict] = {}
    for path in sorted(output_dir.rglob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        suite = summary["task_suite"]
        slot = merged.setdefault(
            suite,
            {"episodes": 0, "successes": 0, "per_category": {}},
        )
        slot["episodes"] += summary["total_episodes"]
        slot["successes"] += summary["total_successes"]
        for category, counts in summary["per_category"].items():
            cat = slot["per_category"].setdefault(category, {"total": 0, "success": 0})
            cat["total"] += counts["total"]
            cat["success"] += counts["success"]
        logging.info("Merged %s", path.name)

    if not merged:
        raise FileNotFoundError(f"No *_summary.json under {output_dir}")

    suites = [s for s in LEADERBOARD_SUITES if s in merged] + [s for s in merged if s not in LEADERBOARD_SUITES]
    header = "suite".ljust(16) + "".join(col.rjust(12) for col in LEADERBOARD_COLUMNS) + "Total".rjust(12)
    logging.info(header)
    column_rates = {col: [] for col in LEADERBOARD_COLUMNS}
    suite_rates = []
    for suite in suites:
        slot = merged[suite]
        cells = []
        for col in LEADERBOARD_COLUMNS:
            cat = slot["per_category"].get(col)
            rate = (cat["success"] / cat["total"]) if cat and cat["total"] else float("nan")
            cells.append(rate)
            if cat and cat["total"]:
                column_rates[col].append(rate)
        suite_rate = slot["successes"] / slot["episodes"] if slot["episodes"] else float("nan")
        suite_rates.append(suite_rate)
        logging.info(
            suite.ljust(16)
            + "".join(f"{100.0 * r:11.1f}%" for r in cells)
            + f"{100.0 * suite_rate:11.1f}%"
        )
    mean_cols = []
    for col in LEADERBOARD_COLUMNS:
        rates = column_rates[col]
        mean_cols.append(sum(rates) / len(rates) if rates else float("nan"))
    mean_total = sum(suite_rates) / len(suite_rates)
    logging.info(
        "mean".ljust(16)
        + "".join(f"{100.0 * r:11.1f}%" for r in mean_cols)
        + f"{100.0 * mean_total:11.1f}%"
    )
    out = output_dir / "leaderboard.json"
    out.write_text(
        json.dumps(
            {
                "suites": merged,
                "mean_category_rate": dict(zip(LEADERBOARD_COLUMNS, mean_cols)),  # noqa: B905  (py3.8 client venv)
                "mean_suite_rate": mean_total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tyro.cli(eval_libero)
