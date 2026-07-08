#!/usr/bin/env python3
"""Print episode -> annotated start frame mappings for find9shape."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation_csv",
        default="annotation.csv",
        help="Path to annotation.csv containing episode_index and top_marked_frame_path columns.",
    )
    args = parser.parse_args()

    annotation_csv = Path(args.annotation_csv)
    with annotation_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{annotation_csv} is empty")

    required_columns = {"episode_index", "top_marked_frame_path"}
    missing_columns = required_columns - set(rows[0].keys())
    if missing_columns:
        raise ValueError(f"{annotation_csv} is missing columns: {sorted(missing_columns)}")

    parsed_rows = []
    for row in rows:
        episode_index = int(row["episode_index"])
        top_marked_frame_path = row["top_marked_frame_path"]
        match = re.search(r"episode_\d+_frame_(\d+)_red_point\.png$", top_marked_frame_path)
        if match is None:
            raise ValueError(
                f"Could not parse frame i from top_marked_frame_path for episode "
                f"{episode_index}: {top_marked_frame_path}"
            )
        parsed_rows.append((episode_index, int(match.group(1))))

    for episode_index, start_index in sorted(parsed_rows, key=lambda item: item[0]):
        print(f"episode_{episode_index:06d}: i={start_index}")


if __name__ == "__main__":
    main()
