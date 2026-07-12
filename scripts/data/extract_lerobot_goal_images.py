"""Extract one goal image per episode/view from a LeRobot-format dataset.

The output mirrors LeRobot's video layout:

    goal_images/chunk-000/observation.images.top/episode_000000.png
    goal_images/chunk-000/observation.images.wrist/episode_000000.png

By default the goal image is the last decoded frame of each episode video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_path(template: str, episode_index: int, episode_chunk: int, video_key: str | None = None) -> Path:
    kwargs = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return Path(template.format(**kwargs))


def _read_last_frame(video_path: Path) -> np.ndarray:
    try:
        reader = imageio.get_reader(video_path)
        last_frame = None
        try:
            for frame in reader:
                last_frame = frame[..., :3]
        finally:
            reader.close()
        if last_frame is not None:
            return np.asarray(last_frame, dtype=np.uint8)
    except Exception:
        pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ok, frame_bgr = capture.read()
            if ok:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
        finally:
            capture.release()
    except Exception:
        pass

    return _read_last_frame_with_ffmpeg(video_path)


def _read_last_frame_with_ffmpeg(video_path: Path) -> np.ndarray:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = int(stream.get("nb_frames") or 0)

    select_filter = f"select=eq(n\\,{max(nb_frames - 1, 0)})" if nb_frames > 0 else "select=gte(n\\,0)"
    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        select_filter,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    decoded = subprocess.run(decode_cmd, check=True, capture_output=True)
    expected = height * width * 3
    if len(decoded.stdout) != expected:
        raise RuntimeError(
            f"Decoded last frame has unexpected size for {video_path}: "
            f"bytes={len(decoded.stdout)}, expected={expected}"
        )
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(height, width, 3)


def extract_goal_images(dataset_root: Path, output_root: Path, overwrite: bool = False) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    video_keys = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    ]
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info["chunks_size"])
    video_template = info["video_path"]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(dataset_root),
        "source": "last_frame",
        "total_episodes": total_episodes,
        "video_keys": video_keys,
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for episode_index in tqdm(range(total_episodes), desc="Extracting goal images"):
        episode_chunk = _episode_chunk(episode_index, chunks_size)
        for video_key in video_keys:
            video_rel = _format_path(video_template, episode_index, episode_chunk, video_key)
            video_path = dataset_root / video_rel
            output_path = (
                output_root
                / f"chunk-{episode_chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.png"
            )
            if output_path.exists() and not overwrite:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame = _read_last_frame(video_path)
            Image.fromarray(frame).save(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output goal image root. Defaults to DATASET_ROOT/goal_images.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_root or (args.dataset_root / "goal_images")
    extract_goal_images(args.dataset_root, output_root, overwrite=args.overwrite)
    print(f"Goal images written to {output_root}")


if __name__ == "__main__":
    main()
