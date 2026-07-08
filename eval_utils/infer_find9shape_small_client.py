"""
Load one find9shape_small episode, send observations to a websocket policy server, and infer a chunk.

python eval_utils/infer_find9shape_small_client.py \
  --dataset_root ./datasets/find9shape_small \
  --episode_index 0 \
  --start_idx 3 \
  --chunk_size 1 \
  --send_history \
  --host 127.0.0.1 \
  --port 8010

Example:
  python eval_utils/infer_find9shape_small_client.py \
    --dataset_root ./datasets/find9shape_small \
    --episode_index 0 \
    --start_idx 3 \
    --chunk_size 4 \
    --port 8010
"""


import json
import logging
import sys
from pathlib import Path
import argparse
import subprocess

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_dataset_path(template: str, episode_index: int, episode_chunk: int) -> Path:
    return Path(template.format(episode_index=episode_index, episode_chunk=episode_chunk))


def _read_video_frames(path: Path) -> np.ndarray:
    try:
        reader = imageio.get_reader(path)
        try:
            frames = [frame[..., :3] for frame in reader]
        finally:
            reader.close()
    except Exception:
        try:
            frames = _read_video_frames_with_ffmpeg(path)
        except Exception:
            frames = []
    if not frames:
        capture = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _read_video_frames_with_ffmpeg(path: Path) -> list[np.ndarray]:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    decoded = subprocess.run(decode_cmd, check=True, capture_output=True)
    frame_size = height * width * 3
    if len(decoded.stdout) % frame_size != 0:
        raise RuntimeError(
            f"Decoded raw video size is not divisible by frame size for {path}: "
            f"bytes={len(decoded.stdout)}, frame_size={frame_size}"
        )
    num_frames = len(decoded.stdout) // frame_size
    array = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(num_frames, height, width, 3)
    return [frame for frame in array]


def load_find9shape_episode(dataset_root: str, episode_index: int) -> dict:
    root = Path(dataset_root)
    info = _load_info(root)
    episode_chunk = _episode_chunk(episode_index, int(info["chunks_size"]))

    parquet_rel = _format_dataset_path(info["data_path"], episode_index, episode_chunk)
    parquet_path = root / parquet_rel
    df = pd.read_parquet(parquet_path)

    video_template = info["video_path"]
    top_path = root / _format_dataset_path(
        video_template.format(
            episode_index=episode_index,
            episode_chunk=episode_chunk,
            video_key="observation.images.top",
        ),
        episode_index,
        episode_chunk,
    )
    wrist_path = root / _format_dataset_path(
        video_template.format(
            episode_index=episode_index,
            episode_chunk=episode_chunk,
            video_key="observation.images.wrist",
        ),
        episode_index,
        episode_chunk,
    )

    top_frames = _read_video_frames(top_path)
    wrist_frames = _read_video_frames(wrist_path)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32, copy=False)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32, copy=False)
    prompt = str(df["task"].iloc[0]) if "task" in df.columns else ""

    length = min(len(df), len(top_frames), len(wrist_frames), len(states), len(actions))
    return {
        "episode_index": episode_index,
        "length": length,
        "prompt": prompt,
        "top": top_frames[:length],
        "wrist": wrist_frames[:length],
        "states": states[:length],
        "actions": actions[:length],
    }


def main(
    dataset_root: str = "./datasets/find9shape_small",
    episode_index: int = 0,
    start_idx: int = 0,
    chunk_size: int = 4,
    host: str = "127.0.0.1",
    port: int = 8010,
    send_history: bool = False,
    single_block: bool = False,
    block_size: int = 4,
    block_stride: int | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    episode = load_find9shape_episode(dataset_root, episode_index)
    length = int(episode["length"])
    if start_idx < 0 or start_idx >= length:
        raise ValueError(f"start_idx must be in [0, {length - 1}], got {start_idx}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_stride is None:
        block_stride = block_size
    if block_stride < 1:
        raise ValueError(f"block_stride must be positive, got {block_stride}")
    end_idx = min(start_idx + (block_size if single_block else chunk_size), length)
    if end_idx <= start_idx:
        raise ValueError(f"Empty chunk: start_idx={start_idx}, chunk_size={chunk_size}, length={length}")

    from eval_utils.policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=port)
    session_id = f"find9shape_ep{episode_index:06d}_idx{start_idx:06d}"
    client.reset({"session_id": session_id})

    print(f"Loaded episode {episode_index}: length={length}, prompt={episode['prompt']!r}")
    print(f"Inferring frames [{start_idx}, {end_idx}) via {host}:{port}")

    last_prediction = None
    if single_block:
        request_indices = range(start_idx, min(start_idx + chunk_size * block_stride, length), block_stride)
    else:
        request_indices = range(start_idx, end_idx)

    for local_i, frame_idx in enumerate(request_indices):
        if single_block:
            block_end = min(frame_idx + block_size, length)
            top = np.stack(episode["top"][frame_idx:block_end], axis=0)
            wrist = np.stack(episode["wrist"][frame_idx:block_end], axis=0)
            if len(top) < block_size:
                pad_top = np.repeat(top[-1:],
                                    block_size - len(top),
                                    axis=0)
                pad_wrist = np.repeat(wrist[-1:],
                                      block_size - len(wrist),
                                      axis=0)
                top = np.concatenate([top, pad_top], axis=0)
                wrist = np.concatenate([wrist, pad_wrist], axis=0)
            request_desc = f"block=[{frame_idx},{block_end}) sent_shape={tuple(top.shape)}"
        elif send_history:
            top = np.stack(episode["top"][start_idx : frame_idx + 1], axis=0)
            wrist = np.stack(episode["wrist"][start_idx : frame_idx + 1], axis=0)
            request_desc = f"history=[{start_idx},{frame_idx + 1}) sent_shape={tuple(top.shape)}"
        else:
            top = episode["top"][frame_idx]
            wrist = episode["wrist"][frame_idx]
            request_desc = f"frame={frame_idx} sent_shape={tuple(np.asarray(top).shape)}"
        obs = {
            "observation/images/top": top,
            "observation/images/wrist": wrist,
            "observation/state": episode["states"][frame_idx],
            "prompt": episode["prompt"],
            "session_id": session_id,
        }
        pred = np.asarray(client.infer(obs), dtype=np.float32)
        last_prediction = pred

        gt = episode["actions"][frame_idx : min(frame_idx + pred.shape[0], length)]
        comparable = min(len(gt), len(pred))
        mae = float(np.mean(np.abs(pred[:comparable] - gt[:comparable]))) if comparable else float("nan")
        print(
            f"request={local_i} frame_idx={frame_idx} pred_shape={tuple(pred.shape)} "
            f"gt_shape={tuple(gt.shape)} mae={mae:.6f} {request_desc}"
        )
        print("pred:")
        print(np.array2string(pred, precision=5, suppress_small=False))
        print("gt:")
        print(np.array2string(gt, precision=5, suppress_small=False))

    client.reset({"session_id": session_id})
    if last_prediction is not None:
        print(f"Last predicted action chunk shape: {tuple(last_prediction.shape)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="./datasets/find9shape_small")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--send_history", action="store_true")
    parser.add_argument(
        "--single_block",
        action="store_true",
        help="Send one video block per request instead of one frame per request.",
    )
    parser.add_argument("--block_size", type=int, default=4, help="Number of frames to send per block request.")
    parser.add_argument(
        "--block_stride",
        type=int,
        default=None,
        help="Frame stride between block requests; defaults to block_size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        start_idx=args.start_idx,
        chunk_size=args.chunk_size,
        host=args.host,
        port=args.port,
        send_history=args.send_history,
        single_block=args.single_block,
        block_size=args.block_size,
        block_stride=args.block_stride,
    )
