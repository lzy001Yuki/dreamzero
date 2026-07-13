"""
Run inference for a find9shape annotation-context checkpoint.

The client reads annotation.csv, parses the marked frame i for the selected
episode, sends a fixed video block that matches annotation-context training, and
uses state/action targets starting at i.

Default mode is train_window:
  - video block: absolute episode frames [0, block_size)
  - state: states[i]
  - ground-truth comparison: actions[i:i + pred_horizon]

This mirrors ShardedLeRobotAnnotationContextDataset, where video indices are not
shifted by i, while state/action are anchored at i.

Example:
  python eval_utils/infer_find9shape_annotation_client.py \
    --dataset_root ./datasets/find9shape_small \
    --annotation_csv annotation.csv \
    --episode_index 0 \
    --block_size 17 \
    --port 8010
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.infer_find9shape_small_client import load_find9shape_episode
from eval_utils.policy_client import WebsocketClientPolicy


logger = logging.getLogger(__name__)


def _load_annotation_indices(annotation_csv: str | Path) -> dict[int, int]:
    path = Path(annotation_csv)
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        required = {"episode_index", "top_marked_frame_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        mapping: dict[int, int] = {}
        for row in reader:
            episode_index = int(row["episode_index"])
            marked_path = row["top_marked_frame_path"]
            match = re.search(r"episode_\d+_frame_(\d+)_red_point\.png$", marked_path)
            if match is None:
                raise ValueError(
                    f"Could not parse annotation frame from top_marked_frame_path "
                    f"for episode {episode_index}: {marked_path}"
                )
            frame_index = int(match.group(1))
            if episode_index in mapping and mapping[episode_index] != frame_index:
                raise ValueError(
                    f"episode_index={episode_index} has conflicting annotation frames: "
                    f"{mapping[episode_index]} and {frame_index}"
                )
            mapping[episode_index] = frame_index
    return mapping


def _pad_or_trim_video(frames: np.ndarray, block_size: int, pad_side: str = "right") -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Expected video block with shape (T,H,W,C), got {frames.shape}")
    if len(frames) >= block_size:
        return frames[:block_size].astype(np.uint8, copy=False)
    if len(frames) == 0:
        raise ValueError("Cannot pad an empty video block")
    pad_count = block_size - len(frames)
    pad_frame = frames[:1] if pad_side == "left" else frames[-1:]
    padding = np.repeat(pad_frame, pad_count, axis=0)
    if pad_side == "left":
        frames = np.concatenate([padding, frames], axis=0)
    else:
        frames = np.concatenate([frames, padding], axis=0)
    return frames.astype(np.uint8, copy=False)


def _select_video_block(
    episode: dict,
    annotation_idx: int,
    block_size: int,
    context_mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    length = int(episode["length"])
    if context_mode == "through_i":
        start = 0
        end = annotation_idx + 1
        top = np.asarray(episode["top"][start:end]).astype(np.uint8, copy=False)
        wrist = np.asarray(episode["wrist"][start:end]).astype(np.uint8, copy=False)
        desc = f"{context_mode} raw=[{start},{end}) sent_shape={tuple(top.shape)}"
        return top, wrist, desc
    if context_mode == "train_window":
        start = 0
        end = min(block_size, length)
        pad_side = "right"
    elif context_mode == "from_start_to_i":
        start = 0
        end = annotation_idx + 1
        pad_side = "right"
    elif context_mode == "before_i":
        start = max(0, annotation_idx - block_size + 1)
        end = annotation_idx + 1
        pad_side = "left"
    elif context_mode == "after_i":
        start = annotation_idx
        end = min(annotation_idx + block_size, length)
        pad_side = "right"
    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")

    top = _pad_or_trim_video(np.asarray(episode["top"][start:end]), block_size, pad_side=pad_side)
    wrist = _pad_or_trim_video(np.asarray(episode["wrist"][start:end]), block_size, pad_side=pad_side)
    desc = f"{context_mode} raw=[{start},{end}) padded_shape={tuple(top.shape)}"
    return top, wrist, desc


def main(
    dataset_root: str = "./datasets/find9shape_small",
    annotation_csv: str = "annotation.csv",
    episode_index: int = 0,
    block_size: int = 17,
    context_mode: str = "train_window",
    host: str = "127.0.0.1",
    port: int = 8010,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    annotation_indices = _load_annotation_indices(annotation_csv)
    if episode_index not in annotation_indices:
        raise KeyError(f"episode_index={episode_index} not found in {annotation_csv}")

    episode = load_find9shape_episode(dataset_root, episode_index)
    length = int(episode["length"])
    annotation_idx = int(annotation_indices[episode_index])
    if annotation_idx < 0 or annotation_idx >= length:
        raise ValueError(
            f"Annotation frame i={annotation_idx} is outside episode length {length} "
            f"for episode {episode_index}"
        )

    top, wrist, request_desc = _select_video_block(episode, annotation_idx, block_size, context_mode)
    session_id = f"find9shape_annotation_ep{episode_index:06d}_idx{annotation_idx:06d}"

    client = WebsocketClientPolicy(host=host, port=port)
    client.reset({"session_id": session_id})

    obs = {
        "observation/images/top": top,
        "observation/images/wrist": wrist,
        "observation/state": episode["states"][annotation_idx],
        "prompt": episode["prompt"],
        "session_id": session_id,
    }

    print(
        f"Loaded episode {episode_index}: length={length}, annotation_i={annotation_idx}, "
        f"prompt={episode['prompt']!r}"
    )
    print(f"Sending annotation context block via {host}:{port}: {request_desc}")
    pred = np.asarray(client.infer(obs), dtype=np.float32)

    gt = episode["actions"][annotation_idx : min(annotation_idx + pred.shape[0], length)]
    comparable = min(len(gt), len(pred))
    mae = float(np.mean(np.abs(pred[:comparable] - gt[:comparable]))) if comparable else float("nan")
    print(
        f"pred_shape={tuple(pred.shape)} gt_shape={tuple(gt.shape)} "
        f"gt_range=[{annotation_idx},{annotation_idx + comparable}) mae={mae:.6f}"
    )
    print("pred:")
    print(np.array2string(pred, precision=5, suppress_small=False))
    print("gt:")
    print(np.array2string(gt, precision=5, suppress_small=False))

    client.reset({"session_id": session_id})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="./datasets/find9shape_small")
    parser.add_argument("--annotation_csv", default="annotation.csv")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=17)
    parser.add_argument(
        "--context_mode",
        choices=("train_window", "from_start_to_i", "before_i", "after_i", "through_i"),
        default="train_window",
        help=(
            "train_window matches the annotation-context dataset: video uses absolute "
            "[0, block_size), while state/action are anchored at annotation i. "
            "through_i sends raw frames [0, i + 1) without padding or trimming."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset_root=args.dataset_root,
        annotation_csv=args.annotation_csv,
        episode_index=args.episode_index,
        block_size=args.block_size,
        context_mode=args.context_mode,
        host=args.host,
        port=args.port,
    )
