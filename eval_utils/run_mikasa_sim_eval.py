"""Run closed-loop inference in a MIKASA-Robo-VLA simulation environment.

This script is a small adapter between the canonical MIKASA observation/action
contract and the websocket policy server used by the DreamZero eval utilities.

Examples:
  # Smoke-test the MIKASA environment without a policy server.
  python eval_utils/run_mikasa_sim_eval.py --dry-run --episodes 1 --sim-backend cpu

  # Roll out a hosted policy server.
  python eval_utils/run_mikasa_sim_eval.py \
    --env-id RememberColor3-VLA-v0 --host localhost --port 8010 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

np = None

REPO_ROOT = Path(__file__).resolve().parents[1]
MIKASA_ROOT = REPO_ROOT / "MIKASA-Robo"
for path in (REPO_ROOT, MIKASA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass
class EpisodeStats:
    episode: int
    seed: int
    success_once: bool
    episode_return: float
    n_steps: int
    prompt: str
    video_path: str | None


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_scalar(value: Any, default: Any = None) -> Any:
    import torch

    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.detach().reshape(-1)[0].cpu().item()
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _extract_language_instruction(info: dict[str, Any], fallback: str = "") -> str:
    import torch

    language = info.get("language_instruction", fallback)
    if isinstance(language, str):
        return language
    if isinstance(language, bytes):
        return language.decode("utf-8")
    if torch.is_tensor(language):
        language = language.detach().cpu().numpy()
    if isinstance(language, np.ndarray):
        if language.size == 0:
            return fallback
        language = language.reshape(-1)[0]
        return _extract_language_instruction({"language_instruction": language}, fallback)
    if isinstance(language, (list, tuple)):
        if not language:
            return fallback
        return _extract_language_instruction({"language_instruction": language[0]}, fallback)
    return str(language) if language is not None else fallback


def _camera_frames(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return top and wrist frames from canonical MIKASA obs."""
    if "rgb" not in obs:
        raise KeyError("Expected canonical MIKASA obs to contain obs['rgb'].")
    rgb = _to_numpy(obs["rgb"])
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 6:
        raise ValueError(f"Expected obs['rgb'] shape (1, H, W, 6), got {rgb.shape}.")
    top = rgb[0, :, :, :3].astype(np.uint8, copy=False)
    wrist = rgb[0, :, :, 3:6].astype(np.uint8, copy=False)
    return top, wrist


def _proprio(obs: dict[str, Any]) -> np.ndarray:
    if "proprio" not in obs:
        raise KeyError("Expected canonical MIKASA obs to contain obs['proprio'].")
    proprio = _to_numpy(obs["proprio"]).astype(np.float32, copy=False)
    if proprio.ndim == 2:
        proprio = proprio[0]
    if proprio.shape != (7,):
        raise ValueError(f"Expected obs['proprio'] shape (7,) or (1, 7), got {proprio.shape}.")
    return proprio


def _make_viz_frame(obs: dict[str, Any]) -> np.ndarray:
    top, wrist = _camera_frames(obs)
    return np.concatenate([top, wrist], axis=1)


def _normalise_action(action: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 1:
        raise ValueError(f"Expected one action vector, got shape {action.shape}.")

    # DreamZero DROID-style servers may return 8D actions: 7 movement/joint
    # values plus a [0, 1] gripper position. MIKASA expects 7D EE-delta action:
    # 6 movement values plus gripper command in [-1, 1].
    if action.shape[-1] == 8 and action_shape == (7,):
        gripper = 1.0 if action[-1] > 0.5 else -1.0
        action = np.concatenate([action[:6], np.array([gripper], dtype=np.float32)])

    if tuple(action.shape) != action_shape:
        raise ValueError(f"Expected action shape {action_shape}, got {tuple(action.shape)}.")
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


class MikasaPolicyClient:
    """Chunked policy adapter for MIKASA's top/wrist/proprio observation."""

    def __init__(self, host: str, port: int, open_loop_horizon: int) -> None:
        from eval_utils.policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        self.open_loop_horizon = int(open_loop_horizon)
        self.session_id = str(uuid.uuid4())
        self._actions: np.ndarray | None = None
        self._cursor = 0

    def reset(self) -> None:
        self.session_id = str(uuid.uuid4())
        self._actions = None
        self._cursor = 0
        try:
            self.client.reset({"session_id": self.session_id})
        except Exception:
            # Some lightweight policy servers do not implement reset. A new
            # session_id is still enough for the DreamZero adapters in this repo.
            pass

    def infer(self, obs: dict[str, Any], prompt: str, action_shape: tuple[int, ...]) -> np.ndarray:
        if self._actions is None or self._cursor >= min(len(self._actions), self.open_loop_horizon):
            top, wrist = _camera_frames(obs)
            proprio = _proprio(obs)
            request = {
                "observation/images/top": top,
                "observation/images/wrist": wrist,
                "observation/state": proprio,
                "prompt": prompt,
                "session_id": self.session_id,
            }
            response = self.client.infer(request)
            actions = response.get("actions") if isinstance(response, dict) else response
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(f"Policy server returned invalid action chunk shape {actions.shape}.")
            self._actions = actions
            self._cursor = 0

        action = _normalise_action(self._actions[self._cursor], action_shape)
        self._cursor += 1
        return action


class RandomMikasaPolicy:
    """Dry-run policy that samples valid MIKASA actions."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def infer(self, obs: dict[str, Any], prompt: str, action_shape: tuple[int, ...]) -> np.ndarray:
        del obs, prompt
        return self.rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)


def make_env(args: argparse.Namespace):
    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401, registers env IDs
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        reward_mode=args.reward_mode,
        render_mode="all",
        sim_backend=args.sim_backend,
    )
    return apply_mikasa_vla_wrappers(env, include_overlays=args.include_overlays)


def run_episode(
    env,
    policy: MikasaPolicyClient | RandomMikasaPolicy,
    *,
    episode: int,
    seed: int,
    prompt_override: str | None,
    save_video: bool,
    video_dir: Path,
) -> EpisodeStats:
    import mediapy
    import torch
    from tqdm import tqdm

    obs, info = env.reset(seed=seed)
    policy.reset()

    prompt = prompt_override or _extract_language_instruction(info)
    action_shape = tuple(env.action_space.shape)
    max_steps = int(getattr(env, "max_episode_steps"))
    frames: list[np.ndarray] = []
    success_once = False
    episode_return = 0.0
    n_steps = 0

    for _ in tqdm(range(max_steps), desc=f"episode {episode}", leave=False):
        if save_video:
            frames.append(_make_viz_frame(obs))

        action_np = policy.infer(obs, prompt, action_shape)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)

        n_steps += 1
        episode_return += float(_first_scalar(reward, 0.0))
        success_once = success_once or bool(_first_scalar(info.get("success"), False))
        done = bool(_first_scalar(terminated, False)) or bool(_first_scalar(truncated, False))
        if done:
            if save_video:
                frames.append(_make_viz_frame(obs))
            break

    video_path = None
    if save_video and frames:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / f"episode_{episode:04d}.mp4")
        mediapy.write_video(video_path, frames, fps=30)

    return EpisodeStats(
        episode=episode,
        seed=seed,
        success_once=success_once,
        episode_return=episode_return,
        n_steps=n_steps,
        prompt=prompt,
        video_path=video_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="RememberColor3-VLA-v0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--sim-backend", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--reward-mode", default="normalized_dense")
    parser.add_argument("--prompt", default=None, help="Override info['language_instruction'].")
    parser.add_argument("--include-overlays", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use random actions instead of a websocket server.")
    parser.add_argument("--no-video", action="store_true", help="Do not write rollout videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "mikasa_sim_eval",
        help="Directory for videos and summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    global np
    args = parse_args()

    try:
        import numpy as np  # noqa: PLW0603
        import gymnasium  # noqa: F401
        import mediapy  # noqa: F401
        import torch  # noqa: F401
        import tqdm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc.name}. Activate/install the project "
            "environment before running MIKASA sim evaluation."
        ) from exc

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive.")

    run_dir = args.output_dir / args.env_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env = make_env(args)
    policy: MikasaPolicyClient | RandomMikasaPolicy
    if args.dry_run:
        policy = RandomMikasaPolicy(seed=args.start_seed)
    else:
        policy = MikasaPolicyClient(args.host, args.port, args.open_loop_horizon)

    stats: list[EpisodeStats] = []
    try:
        for episode in range(args.episodes):
            stats.append(
                run_episode(
                    env,
                    policy,
                    episode=episode,
                    seed=args.start_seed + episode,
                    prompt_override=args.prompt,
                    save_video=not args.no_video,
                    video_dir=run_dir / "videos",
                )
            )
    finally:
        env.close()

    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "env_id": args.env_id,
        "episodes": [asdict(item) for item in stats],
        "success_rate": float(np.mean([item.success_once for item in stats])) if stats else 0.0,
        "mean_return": float(np.mean([item.episode_return for item in stats])) if stats else 0.0,
        "dry_run": bool(args.dry_run),
        "sim_backend": args.sim_backend,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Saved MIKASA sim eval to {run_dir}")
    print(f"success_rate={summary['success_rate']:.3f}, mean_return={summary['mean_return']:.3f}")


if __name__ == "__main__":
    main()
