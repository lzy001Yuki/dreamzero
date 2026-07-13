"""
Serve a find9shape annotation-context DreamZero checkpoint.

This is a thin wrapper around serve_find9shape_small.py with defaults that match
the annotation-context training setup: the first request should carry a full
context video block, and the server should use that whole block instead of
collapsing the first causal call to a single frame.

Example:
  python eval_utils/serve_find9shape_annotation.py \
    --model_path /path/to/annotation/checkpoint \
    --port 8010 \
    --save_video_pred
"""

from __future__ import annotations

import argparse

from eval_utils.serve_find9shape_small import DEFAULT_FRAMES_PER_CHUNK, main as serve_find9shape


ANNOTATION_FRAMES_PER_CHUNK = 17


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument(
        "--frames_per_chunk",
        type=int,
        default=ANNOTATION_FRAMES_PER_CHUNK,
        help=(
            "Number of video frames to feed on the first annotation request. "
            f"Defaults to {ANNOTATION_FRAMES_PER_CHUNK}; the base find9shape default is "
            f"{DEFAULT_FRAMES_PER_CHUNK}."
        ),
    )
    parser.add_argument("--non_causal", action="store_true", help="Use lazy_joint_forward instead of causal inference.")
    parser.add_argument("--save_video_pred", action="store_true", help="Decode and save returned video_pred mp4 files.")
    parser.add_argument(
        "--video_output_dir",
        default="./video_pred_output/find9shape_annotation",
        help="Base directory for decoded video_pred mp4 files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Exact directory to save decoded video_pred mp4 files. Implies --save_video_pred.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    serve_find9shape(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        tokenizer_path=args.tokenizer_path,
        image_height=args.image_height,
        image_width=args.image_width,
        frames_per_chunk=args.frames_per_chunk,
        causal=not args.non_causal,
        first_call_full_video=True,
        save_video_pred=args.save_video_pred or args.output_dir is not None,
        video_output_dir=args.video_output_dir,
        output_dir=args.output_dir,
    )
