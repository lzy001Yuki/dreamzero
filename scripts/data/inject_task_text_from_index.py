"""
Inject a string task column into LeRobot parquet files from task_index.

DreamZero's find9shape/real_panda_single_arm config reads language from
annotation.task -> original_key "task". Some LeRobot datasets only store
task_index in parquet and keep task text in meta/tasks.jsonl. This utility
expands task_index into a per-row task string column without changing row
counts.

Usage:
  python scripts/data/inject_task_text_from_index.py \
      --dataset-path /path/to/find9shape_dataset

  # Copy to a new dataset directory first:
  python scripts/data/inject_task_text_from_index.py \
      --dataset-path /path/to/raw_find9shape \
      --output-path /path/to/find9shape_with_task
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_tasks(tasks_path: Path) -> dict[int, str]:
    if not tasks_path.exists():
        log.error("tasks.jsonl not found at %s", tasks_path)
        sys.exit(1)

    task_map: dict[int, str] = {}
    with open(tasks_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "task_index" not in item or "task" not in item:
                raise ValueError(f"{tasks_path}:{line_no} must contain task_index and task")
            task_map[int(item["task_index"])] = str(item["task"])

    if not task_map:
        raise ValueError(f"No tasks found in {tasks_path}")
    return task_map


def get_parquet_paths(dataset_path: Path) -> list[Path]:
    paths = sorted(dataset_path.glob("data/*/episode_*.parquet"))
    if not paths:
        log.error("No parquet files found under %s/data/*/episode_*.parquet", dataset_path)
        sys.exit(1)
    return paths


def prepare_output_dataset(dataset_path: Path, output_path: Path | None, force: bool) -> Path:
    if output_path is None:
        return dataset_path

    output_path = output_path.resolve()
    if output_path == dataset_path:
        return dataset_path

    if output_path.exists():
        if not force:
            log.error("Output path already exists: %s. Use --force to overwrite.", output_path)
            sys.exit(1)
        shutil.rmtree(output_path)

    log.info("Copying dataset from %s to %s", dataset_path, output_path)
    shutil.copytree(dataset_path, output_path)
    return output_path


def inject_task_column(
    dataset_path: Path,
    task_index_column: str,
    task_column: str,
    overwrite_task_column: bool,
) -> None:
    task_map = load_tasks(dataset_path / "meta" / "tasks.jsonl")
    parquet_paths = get_parquet_paths(dataset_path)

    log.info("Loaded %d task(s) from meta/tasks.jsonl", len(task_map))
    log.info("Updating %d parquet file(s)", len(parquet_paths))

    for parquet_path in tqdm(parquet_paths, desc="Injecting task text"):
        df = pd.read_parquet(parquet_path)

        if task_index_column not in df.columns:
            raise KeyError(f"{parquet_path} does not contain column {task_index_column!r}")

        if task_column in df.columns and not overwrite_task_column:
            log.info("%s already has %r; skipping", parquet_path, task_column)
            continue

        missing_indices = sorted(set(int(v) for v in df[task_index_column].dropna().unique()) - set(task_map))
        if missing_indices:
            raise KeyError(
                f"{parquet_path} has task_index values missing from tasks.jsonl: {missing_indices}"
            )

        row_count_before = len(df)
        df[task_column] = df[task_index_column].map(lambda v: task_map[int(v)])
        if len(df) != row_count_before:
            raise RuntimeError(f"Row count changed unexpectedly for {parquet_path}")

        df.to_parquet(parquet_path, index=False)

    log.info("Done. Parquet row counts were preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject parquet task text from task_index using meta/tasks.jsonl.",
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="LeRobot/DreamZero dataset root")
    parser.add_argument("--output-path", type=str, default=None, help="Copy dataset here before editing")
    parser.add_argument("--task-index-column", type=str, default="task_index")
    parser.add_argument("--task-column", type=str, default="task")
    parser.add_argument("--overwrite-task-column", action="store_true", help="Overwrite existing task column")
    parser.add_argument("--force", action="store_true", help="Overwrite --output-path if it exists")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        log.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    output_path = Path(args.output_path).resolve() if args.output_path else None
    target_path = prepare_output_dataset(dataset_path, output_path, args.force)
    inject_task_column(
        target_path,
        task_index_column=args.task_index_column,
        task_column=args.task_column,
        overwrite_task_column=args.overwrite_task_column,
    )


if __name__ == "__main__":
    main()
