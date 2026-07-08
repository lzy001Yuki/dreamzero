"""
Serve a DreamZero policy trained for datasets/find9shape_small over the websocket policy server.

The websocket response remains the predicted action chunk. If --save_video_pred is enabled,
the server also decodes returned video_pred latents and writes mp4 files to --video_output_dir
when the client sends reset or the session changes.
python eval_utils/serve_find9shape_small.py \
  --model_path /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero/checkpoints/dreamzero_find9shape_lora/checkpoint-8000 \
  --port 8010 \
  --save_video_pred \
  --video_output_dir ./video_pred_output/find9shape_small
Client observation keys:
  - observation/images/top: uint8 image or video, shape (H, W, 3) or (T, H, W, 3)
  - observation/images/wrist: uint8 image or video, shape (H, W, 3) or (T, H, W, 3)
  - observation/state: float array, shape (7,) or (1, 7), [eef_pose(6), gripper(1)]
  - prompt: task string
  - session_id: optional episode/session id

Model Batch keys:
  - video.top
  - video.wrist
  - state.eef_pose
  - state.gripper
  - annotation.task

Response:
  - action chunk as float32 ndarray, shape (N, 7), [eef_delta(6), gripper(1)]
python eval_utils/serve_find9shape_small.py --model_path /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero/checkpoints/dreamzero_find9shape_lora/checkpoint-8000 --port 8010 2>&1 | tee lo
python eval_utils/serve_find9shape_small.py --model_path /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/tool_adaptation/world2action/dreamzero/checkpoints/DreamZero-AgiBot 
  另一个终端运行 client：
Example:
  python eval_utils/serve_find9shape_small.py \
    --model_path ./checkpoints/find9shape_small \
    --port 8010 \
    --save_video_pred \
    --video_output_dir ./video_pred_output/find9shape_small
"""

import logging
import os
import sys
from pathlib import Path
import argparse
import datetime

import cv2
from einops import rearrange
import imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch

# Avoid FailOnRecompileLimitHit when serving: the flow scheduler's torch.compile'd
# multistep_uni_p_bh_update can recompile across inference steps/sessions under
# fullgraph=True, dynamic=False. Increase limits so the server does not hit the
# default cap during normal closed-loop use.
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi_client.base_policy import BasePolicy

from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


logger = logging.getLogger(__name__)

DEFAULT_IMAGE_HEIGHT = 128
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_FRAMES_PER_CHUNK = 4

VIDEO_KEY_MAPPING = {
    "observation/images/top": "video.top",
    "observation/images/wrist": "video.wrist",
}


def _maybe_init_distributed(device: str) -> None:
    """Initialize torch distributed because GrootSimPolicy expects a device mesh."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    if device == "cuda":
        torch.cuda.set_device(0)


def _get_expected_video_resolution(policy: GrootSimPolicy) -> tuple[int, int]:
    """Return (height, width) expected by VideoToTensor before resize/crop transforms."""
    eval_transform = getattr(policy, "eval_transform", None)
    if not isinstance(eval_transform, ComposedModalityTransform):
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    for transform in eval_transform.transforms:
        original_resolutions = getattr(transform, "original_resolutions", None)
        if original_resolutions:
            width, height = next(iter(original_resolutions.values()))
            return (int(height), int(width))
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def _resize_frame_or_video(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a frame/video to the pre-transform resolution expected by the checkpoint."""
    frames = np.asarray(frames)
    if frames.ndim == 3:
        if frames.shape[:2] != (target_h, target_w):
            frames = cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frames.astype(np.uint8, copy=False)
    if frames.ndim != 4:
        raise ValueError(f"Expected image/video with 3 or 4 dims, got shape {frames.shape}")
    if frames.shape[1:3] == (target_h, target_w):
        return frames.astype(np.uint8, copy=False)
    return np.stack(
        [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for frame in frames],
        axis=0,
    ).astype(np.uint8, copy=False)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_action_value(action_batch, key: str):
    if isinstance(action_batch, dict):
        return action_batch.get(key)
    if hasattr(action_batch, key):
        return getattr(action_batch, key)
    state = action_batch.__getstate__() if hasattr(action_batch, "__getstate__") else {}
    if isinstance(state, dict):
        return state.get(key)
    return None


class Find9ShapePolicy(BasePolicy):
    """Adapter from websocket observations to the real_panda_single_arm DreamZero policy."""

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        image_height: int,
        image_width: int,
        frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
        causal: bool = True,
        first_call_full_video: bool = False,
        output_dir: str | None = None,
    ) -> None:
        self._policy = groot_policy
        self._image_height = image_height
        self._image_width = image_width
        self._frames_per_chunk = frames_per_chunk
        self._causal = causal
        self._first_call_full_video = first_call_full_video
        self._output_dir = output_dir
        self._video_preds = []
        self._frame_buffers = {model_key: [] for model_key in VIDEO_KEY_MAPPING.values()}
        self._is_first_call = True
        self._current_session_id = None

    def _convert_observation(self, obs: dict) -> dict:
        received_lengths = {}
        for client_key, model_key in VIDEO_KEY_MAPPING.items():
            if client_key not in obs:
                continue
            frames = _resize_frame_or_video(obs[client_key], self._image_height, self._image_width)
            if frames.ndim == 3:
                self._frame_buffers[model_key].append(frames)
                received_lengths[model_key] = 1
            else:
                self._frame_buffers[model_key].extend(list(frames))
                received_lengths[model_key] = int(frames.shape[0])

        if self._is_first_call and self._causal and not self._first_call_full_video:
            num_frames = 1
        elif self._is_first_call and self._causal and received_lengths:
            num_frames = min(max(received_lengths.values()), self._frames_per_chunk)
        else:
            num_frames = self._frames_per_chunk

        converted = {}
        for model_key, buffer in self._frame_buffers.items():
            if not buffer:
                converted[model_key] = np.zeros(
                    (num_frames, self._image_height, self._image_width, 3), dtype=np.uint8
                )
                continue
            frames_to_use = buffer[-num_frames:]
            while len(frames_to_use) < num_frames:
                frames_to_use.insert(0, frames_to_use[0])
            converted[model_key] = np.stack(frames_to_use, axis=0).astype(np.uint8, copy=False)

        state = np.asarray(obs.get("observation/state", np.zeros(7, dtype=np.float32)), dtype=np.float64)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if state.shape[-1] != 7:
            raise ValueError(f"find9shape_small state must have 7 dims, got shape {state.shape}")
        converted["state.eef_pose"] = state[..., :6]
        converted["state.gripper"] = state[..., 6:7]
        converted["annotation.task"] = obs.get("prompt", "")
        return converted

    def _convert_action(self, action_batch) -> np.ndarray:
        eef_delta = _get_action_value(action_batch, "action.eef_delta")
        gripper = _get_action_value(action_batch, "action.gripper")
        if eef_delta is None:
            raise RuntimeError(f"Policy output does not contain action.eef_delta: {action_batch}")
        eef_delta = _as_numpy(eef_delta)
        if eef_delta.ndim == 1:
            eef_delta = eef_delta.reshape(1, -1)
        if eef_delta.shape[-1] != 6:
            raise RuntimeError(f"action.eef_delta must have 6 dims, got {eef_delta.shape}")
        if gripper is None:
            gripper = np.zeros((eef_delta.shape[0], 1), dtype=eef_delta.dtype)
        else:
            gripper = _as_numpy(gripper)
            if gripper.ndim == 1:
                gripper = gripper.reshape(-1, 1)
            gripper = gripper[..., :1]
        return np.concatenate([eef_delta, gripper], axis=-1).astype(np.float32, copy=False)

    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        batch = Batch(obs=self._convert_observation(obs))
        with torch.no_grad():
            if self._causal:
                result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
            else:
                result_batch, video_pred = self._policy.lazy_joint_forward(batch)
        if self._output_dir and video_pred is not None:
            self._video_preds.append(video_pred.detach().cpu())
        action = self._convert_action(result_batch.act)
        self._is_first_call = False
        return action

    def _save_video_predictions(self) -> None:
        if not self._output_dir or not self._video_preds:
            return

        action_head = getattr(self._policy.trained_model, "action_head", None)
        if action_head is None or not hasattr(action_head, "vae"):
            logger.warning("Cannot save infer video: action_head/vae is unavailable")
            self._video_preds = []
            return

        try:
            device = next(action_head.parameters()).device
        except (AttributeError, StopIteration):
            try:
                device = next(self._policy.trained_model.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except TypeError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            os.makedirs(self._output_dir, exist_ok=True)
            video_latents = torch.cat(self._video_preds, dim=2).to(device=device)
            with torch.no_grad():
                frames = action_head.vae.decode(
                    video_latents,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            frame_list = [frame for frame in frames]
            if not frame_list:
                return

            sample_frame = frame_list[0]
            if len(sample_frame.shape) != 3 or sample_frame.shape[2] not in (1, 3, 4):
                logger.warning("Cannot save infer video: decoded frame has invalid shape %s", sample_frame.shape)
                return

            all_mp4_files = [f for f in os.listdir(self._output_dir) if f.endswith(".mp4")]
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            session = self._current_session_id or "session"
            safe_session = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(session))
            output_path = os.path.join(
                self._output_dir,
                f"{len(all_mp4_files):06}_{timestamp}_{safe_session}_frames{len(frame_list)}.mp4",
            )
            imageio.mimsave(output_path, frame_list, fps=5, codec="libx264")
            logger.info("Saved infer video to: %s", output_path)
        except Exception as exc:
            logger.warning("Failed to save infer video: %s", exc)
        finally:
            self._video_preds = []

    def reset(self, reset_info: dict) -> None:
        self._save_video_predictions()
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._current_session_id = None
        action_head = getattr(self._policy.trained_model, "action_head", None)
        if hasattr(action_head, "current_start_frame"):
            action_head.current_start_frame = 0


def main(
    model_path: str,
    port: int = 8010,
    host: str = "0.0.0.0",
    tokenizer_path: str | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
    causal: bool = True,
    first_call_full_video: bool = False,
    save_video_pred: bool = False,
    video_output_dir: str = "./video_pred_output/find9shape_small",
    output_dir: str | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _maybe_init_distributed(device)
    device_mesh = init_device_mesh(device, mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading find9shape_small policy from %s", model_path)
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("real_panda_single_arm"),
        model_path=model_path,
        tokenizer_path_override=tokenizer_path,
        device=device,
        device_mesh=device_mesh,
    )
    expected_h, expected_w = _get_expected_video_resolution(groot_policy)
    h = image_height or expected_h
    w = image_width or expected_w
    logger.info("Using pre-transform video resolution: %dx%d (HxW)", h, w)
    resolved_output_dir = output_dir
    if save_video_pred and resolved_output_dir is None:
        checkpoint_name = os.path.basename(model_path.rstrip("/"))
        resolved_output_dir = os.path.join(video_output_dir, checkpoint_name)
    if resolved_output_dir:
        logger.info("Saving decoded infer videos to %s", resolved_output_dir)

    policy = Find9ShapePolicy(
        groot_policy=groot_policy,
        image_height=h,
        image_width=w,
        frames_per_chunk=frames_per_chunk,
        causal=causal,
        first_call_full_video=first_call_full_video,
        output_dir=resolved_output_dir,
    )
    server_config = PolicyServerConfig(
        image_resolution=(h, w),
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="cartesian_position",
    )
    logger.info("Starting find9shape_small policy server on %s:%d", host, port)
    WebsocketPolicyServer(policy=policy, server_config=server_config, host=host, port=port).serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--save_video_pred", action="store_true", help="Decode and save returned video_pred mp4 files.")
    parser.add_argument(
        "--video_output_dir",
        default="./video_pred_output/find9shape_small",
        help="Base directory for decoded video_pred mp4 files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Exact directory to save decoded video_pred mp4 files. Implies --save_video_pred.",
    )
    parser.add_argument("--frames_per_chunk", type=int, default=DEFAULT_FRAMES_PER_CHUNK)
    parser.add_argument("--non_causal", action="store_true", help="Use lazy_joint_forward instead of causal inference.")
    parser.add_argument(
        "--first_call_full_video",
        action="store_true",
        help="For causal inference, keep the full incoming video block on the first request instead of only the last frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        tokenizer_path=args.tokenizer_path,
        image_height=args.image_height,
        image_width=args.image_width,
        frames_per_chunk=args.frames_per_chunk,
        causal=not args.non_causal,
        first_call_full_video=args.first_call_full_video,
        save_video_pred=args.save_video_pred or args.output_dir is not None,
        video_output_dir=args.video_output_dir,
        output_dir=args.output_dir,
    )
