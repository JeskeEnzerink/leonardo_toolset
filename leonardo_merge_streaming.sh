#!/usr/bin/env bash
# Standalone, memory-efficient merger for Leonardo task TIFFs.
# It only concatenates finished uint16 planes; it does not retrain, destripe,
# postprocess, normalize, or change pixel intensities.
#
# Requirements: Python 3.10+, numpy, tifffile; Leonardo task manifests.
# Works with both RUN_DIR/task_0 and RUN_DIR/tasks/task_0 layouts.
# No Leonardo Python-package import or GPU is needed for merging.
#
# Example (Slurm cluster):
#   sbatch leonardo_merge_streaming.sh --run-dir /path/to/run C00
#   sbatch leonardo_merge_streaming.sh --run-dir /path/to/run \
#       --delete-sources-after-verify C00
#   bash leonardo_merge_streaming.sh --run-dir /path/to/run --dry-run
#
# The optional deletion flag removes ONLY individual task-level TIFFs after
# output validation. FAST checks sample each task; FULL compares all pixels.
# It never deletes task directories, chunks, metadata, source Zarr, or output.
# New final TIFF names are dataset/channel stem + output.suffix from the run
# JSON (for example: SAMPLE_C00_marker_destriped.tif). Old names are recognized.
#
# See --help for all arguments. This file may be run via bash or sbatch.

#SBATCH --job-name=leonardo_merge
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=leo_stream_merge-%j.log

# ========================== USER SETTINGS ===========================
# Change these settings here for each Leonardo run. Command-line options
# (e.g. --run-dir, C00, --delete-sources-after-verify) override them.
#
# Existing run (supply --run-dir /absolute/path/to/run):
RUN_DIR=""
#
# Default: merge all discovered channels. Explicit channel arguments override.
CHANNELS=()
#
OUTPUT_PARENT=""       # empty = RUN_DIR/channels
MATCH_PATTERN="*_alpha1_fftwidth5_uint16.tif"
DELETE_SOURCES_AFTER_VERIFY=false  # true: delete task TIFFs after verification
VERIFICATION_MODE="fast"           # fast (sampled) or full (every single plane)
SAMPLE_PLANES_PER_TASK=5           # fast mode: first, quartiles and last for 5
FREE_SPACE_RESERVE_GIB=5
PROGRESS_EVERY=8
#
# Python environment: adjust these for the target cluster, or leave
# CONDA_ENV empty to use the currently active Python environment.
ANACONDA_MODULE="anaconda3"
CONDA_ENV="leonardo"
PYTHON_BIN="python3"
# ======================== END USER SETTINGS =========================

set -euo pipefail

# Cluster environment activation (skip if already in your Python environment).
if [[ -n "$ANACONDA_MODULE" ]]; then module load "$ANACONDA_MODULE"; fi
if [[ -n "$CONDA_ENV" ]]; then source activate "$CONDA_ENV"; fi

export LEONARDO_MERGE_RUN_DIR="$RUN_DIR"
export LEONARDO_MERGE_OUTPUT_PARENT="$OUTPUT_PARENT"
export LEONARDO_MERGE_CHANNELS="${CHANNELS[*]}"
export LEONARDO_MERGE_MATCH="$MATCH_PATTERN"
export LEONARDO_MERGE_CLEANUP="$DELETE_SOURCES_AFTER_VERIFY"
export LEONARDO_MERGE_VERIFY="$VERIFICATION_MODE"
export LEONARDO_MERGE_SAMPLES_PER_TASK="$SAMPLE_PLANES_PER_TASK"
export LEONARDO_MERGE_RESERVE_GIB="$FREE_SPACE_RESERVE_GIB"
export LEONARDO_MERGE_PROGRESS_EVERY="$PROGRESS_EVERY"

"$PYTHON_BIN" -u - "$@" <<'PY'
"""Discover finished Leonardo TIFF parts from manifests and merge safely."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import tifffile


@dataclass(frozen=True)
class Part:
    path: Path
    start: int
    end: int  # inclusive
    stem: str
    variant: str
    width: int
    shape: tuple
    dtype: np.dtype


@dataclass(frozen=True)
class MergePlan:
    channel: str
    parts: tuple
    output: Path
    partial: Path
    shape: tuple
    dtype: np.dtype
    bytes_needed: int
    already_merged: bool


def arguments():
    parser = argparse.ArgumentParser(
        description="Stream-merge Leonardo task TIFFs in Z order (no giant RAM array).",
        epilog=("No channel arguments: merge every channel found in task manifests. "
                "A channel must have all its tasks finished."),
    )
    parser.add_argument("--run-dir", type=Path,
                        default=os.environ.get("LEONARDO_MERGE_RUN_DIR") or None,
                        help="Leonardo run folder containing task_*/manifest.json")
    parser.add_argument("--output-parent", type=Path,
                        default=os.environ.get("LEONARDO_MERGE_OUTPUT_PARENT") or None,
                        help="Output parent (default: RUN_DIR/channels)")
    parser.add_argument("--match", default=os.environ.get(
        "LEONARDO_MERGE_MATCH", "*_alpha1_fftwidth5_uint16.tif"),
                        help="Source filename glob inside each task directory")
    parser.add_argument("--reserve-gib", type=float,
                        default=float(os.environ.get("LEONARDO_MERGE_RESERVE_GIB", "5")),
                        help="Free-space reserve for new outputs (default: 5 GiB)")
    parser.add_argument("--progress-every", type=int,
                        default=int(os.environ.get("LEONARDO_MERGE_PROGRESS_EVERY", "8")),
                        help="Progress message interval in Z planes (default: 8)")
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--delete-sources-after-verify", action="store_true",
                         dest="delete_sources_after_verify",
                         default=os.environ.get("LEONARDO_MERGE_CLEANUP", "false").lower() == "true",
                         help="Delete selected task TIFFs after configured verification")
    cleanup.add_argument("--keep-sources", action="store_false",
                         dest="delete_sources_after_verify",
                         help="Preserve task TIFFs even if cleanup is true in USER SETTINGS")
    parser.add_argument("--verify", choices=("fast", "full"),
                        default=os.environ.get("LEONARDO_MERGE_VERIFY", "fast"),
                        help="fast: sample input/output planes; full: compare every plane")
    parser.add_argument("--samples-per-task", type=int,
                        default=int(os.environ.get("LEONARDO_MERGE_SAMPLES_PER_TASK", "5")),
                        help="Number of evenly distributed sampled planes per task in fast mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate sources, names, and space without writing/deleting")
    parser.add_argument("channels", nargs="*",
                        help="Channel names (e.g. C00 C01); default: every discovered channel")
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("Set RUN_DIR at the top of this file or pass --run-dir PATH")
    if not args.channels:
        args.channels = os.environ.get("LEONARDO_MERGE_CHANNELS", "").split()
    if args.reserve_gib < 0 or not np.isfinite(args.reserve_gib):
        parser.error("--reserve-gib must be a finite nonnegative number")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.samples_per_task < 2:
        parser.error("--samples-per-task must be >= 2")
    if len(set(args.channels)) != len(args.channels):
        parser.error("Channel names must be unique")
    return args


def discover_manifests(run_dir):
    nested_root = run_dir / "tasks"
    legacy_present = any(p.is_dir() and re.fullmatch(r"task_\d+", p.name)
                         for p in run_dir.iterdir())
    nested_present = nested_root.is_dir() and any(
        p.is_dir() and re.fullmatch(r"task_\d+", p.name)
        for p in nested_root.iterdir())
    if legacy_present and nested_present:
        raise ValueError(f"Ambiguous task layout: both {run_dir}/task_* and "
                         f"{nested_root}/task_* exist. Refusing to guess.")
    task_root = nested_root if nested_present or nested_root.is_dir() else run_dir
    task_dirs = sorted(
        (p for p in task_root.iterdir() if p.is_dir() and re.fullmatch(r"task_\d+", p.name)),
        key=lambda path: int(path.name[5:]),
    )
    if not task_dirs:
        raise ValueError(f"No task_0, task_1, ... directories found in {task_root}")
    print(f"Task directory: {task_root}", flush=True)
    groups = defaultdict(list)
    for task_dir in task_dirs:
        manifest_path = task_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest for task directory: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        task_id = int(task_dir.name[5:])
        if int(manifest["task"]) != task_id:
            raise ValueError(f"Task number mismatch: {manifest_path}")
        name = str(manifest["channel_name"])
        if name in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ValueError(f"Unsafe channel name in manifest: {name!r}")
        groups[name].append((task_dir, manifest))
    return dict(groups)


def inspect_part(path, manifest):
    stem = str(manifest["source_stem"])
    match = re.fullmatch(
        rf"{re.escape(stem)}_z(?P<start>\d+)_(?P<end>\d+)_(?P<variant>.+)\.tif",
        path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected source TIFF name for stem {stem!r}: {path.name}")
    start, end = int(match["start"]), int(match["end"])
    requested_start, requested_end = map(int, manifest["task_z"])
    if (start, end + 1) != (requested_start, requested_end):
        raise ValueError(f"TIFF Z range differs from manifest: {path.name}")
    with tifffile.TiffFile(path) as src:
        shape = tuple(map(int, src.series[0].shape))
        dtype = np.dtype(src.series[0].dtype)
        page_count = len(src.pages)
    if len(shape) == 2 and page_count == 1:
        shape = (1, *shape)
    if len(shape) != 3 or shape[0] != end - start + 1 or page_count != shape[0]:
        raise ValueError(f"Unexpected page count/shape in {path}: {shape}, {page_count} pages")
    if dtype != np.dtype("uint16"):
        raise ValueError(f"Expected uint16 output, got {dtype} in {path}")
    if shape[1:] != tuple(map(int, manifest["shape_yx"])):
        raise ValueError(f"XY shape differs from manifest: {path}")
    return Part(path, start, end, stem, match["variant"],
                max(4, len(match["start"]), len(match["end"])), shape, dtype)


def discover_parts(name, manifest_entries, pattern):
    # Each channel declares how many workers/tasks were assigned, independent
    # of any particular run's hard-coded task count (normally four in Leonardo).
    workers = {int(manifest["channel_workers"]) for _, manifest in manifest_entries}
    if len(workers) != 1:
        raise ValueError(f"{name}: inconsistent channel_workers in manifests")
    worker_count = workers.pop()
    indices = [int(manifest["channel_task"]) for _, manifest in manifest_entries]
    if worker_count < 1 or sorted(indices) != list(range(worker_count)):
        raise ValueError(f"{name}: missing/duplicate task manifests: {indices}")

    identities = {
        (tuple(map(int, m["requested_z"])), tuple(map(int, m["shape_yx"])),
         str(m["source_stem"]), str(m["precision"]),
         str(m.get("source_config_sha256")), str(m.get("zarr_path")))
        for _, m in manifest_entries
    }
    if len(identities) != 1:
        raise ValueError(f"{name}: manifests do not belong to one consistent channel/run")
    requested_z, _, _, _, _, _ = next(iter(identities))

    parts = []
    for task_dir, manifest in manifest_entries:
        z_start, z_end = map(int, manifest["task_z"])
        matches = sorted(task_dir.glob(pattern))
        if z_start == z_end:
            if matches:
                raise ValueError(f"Empty task should not contain a matching TIFF: {task_dir}")
            print(f"{name}: empty Z task {manifest['task']} skipped", flush=True)
            continue
        if len(matches) != 1 or not matches[0].is_file() or matches[0].is_symlink():
            raise ValueError(f"{name}: expected one regular finished TIFF in {task_dir}; "
                             f"found {[p.name for p in matches]}")
        parts.append(inspect_part(matches[0], manifest))
    if not parts:
        raise ValueError(f"{name}: no finished TIFF parts found")

    parts.sort(key=lambda part: part.start)
    first = parts[0]
    if (first.start, parts[-1].end + 1) != requested_z:
        raise ValueError(f"{name}: source TIFFs do not cover requested Z range {requested_z}")
    for previous, current in zip(parts, parts[1:]):
        if previous.end + 1 != current.start:
            raise ValueError(f"{name}: Z gap/overlap between {previous.path} and {current.path}")
    for part in parts[1:]:
        if ((part.shape[1:], part.dtype, part.stem, part.variant) !=
                (first.shape[1:], first.dtype, first.stem, first.variant)):
            raise ValueError(f"{name}: incompatible TIFF parts: {part.path}")
    return tuple(parts)


def validate_output(path, shape, dtype, label="OUTPUT TIFF"):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Completed TIFF missing or symlink: {path}")
    with tifffile.TiffFile(path) as merged:
        observed = tuple(map(int, merged.series[0].shape))
        pages = len(merged.pages)
        if len(observed) == 2 and pages == 1:
            observed = (1, *observed)
        if observed != shape or np.dtype(merged.series[0].dtype) != dtype or pages != shape[0]:
            raise ValueError(f"Existing TIFF failed validation: {path}; shape={observed}, pages={pages}")
    print(f"VALIDATED {label}: {path}", flush=True)


def make_plan(name, manifest_entries, pattern, output_parent, suffix):
    parts = discover_parts(name, manifest_entries, pattern)
    first, last = parts[0], parts[-1]
    width = max(part.width for part in parts)
    clean_output = output_parent / name / f"{first.stem}{suffix}.tif"
    previous_bare_output = output_parent / name / f"{first.stem}.tif"
    previous_fullbrain_output = output_parent / name / f"{first.stem}_fullbrain.tif"
    previous_short_output = output_parent / name / (
        f"{first.stem}_fullbrain_z{first.start:0{width}d}_"
        f"{last.end:0{width}d}.tif"
    )
    legacy_output = output_parent / name / (
        f"{first.stem}_fullbrain_z{first.start:0{width}d}_"
        f"{last.end:0{width}d}_{first.variant}.tif"
    )
    # Never duplicate an existing 60-GiB TIFF just because naming changed.
    known_names = (clean_output, previous_bare_output, previous_fullbrain_output,
                   previous_short_output, legacy_output)
    existing = [path for path in known_names if path.exists() or path.is_symlink()]
    if len(existing) > 1:
        raise ValueError(f"Multiple differently named outputs exist for {name}: "
                         f"{existing}. Inspect before proceeding.")
    output = existing[0] if existing else clean_output
    partial = output.with_name(output.stem + ".partial.tif")
    older_partials = (
        previous_bare_output.with_name(previous_bare_output.stem + ".partial.tif"),
        previous_fullbrain_output.with_name(previous_fullbrain_output.stem + ".partial.tif"),
        previous_short_output.with_name(previous_short_output.stem + ".partial.tif"),
        previous_short_output.with_name(previous_short_output.stem + ".streaming.incomplete.tif"),
        legacy_output.with_name(legacy_output.stem + ".streaming.incomplete.tif"),
    )
    shape = (last.end - first.start + 1, *first.shape[1:])
    needed = int(np.prod(shape, dtype=np.int64) * first.dtype.itemsize)
    if output.is_symlink() or partial.is_symlink():
        raise ValueError(f"Refusing symlinked TIFF path: {output} or {partial}")
    already = output.is_file()
    if already:
        validate_output(output, shape, first.dtype)
    else:
        abandoned = [path for path in (partial, *older_partials)
                     if path.exists() or path.is_symlink()]
        if abandoned:
            raise ValueError(f"Previous incomplete merge exists; inspect first: {abandoned}")
    print(f"{name}: {len(parts)} task TIFFs; shape={shape}; "
          f"new output size={needed / 2**30:.2f} GiB; existing={already}", flush=True)
    return MergePlan(name, parts, output, partial, shape, first.dtype, needed, already)


def merge_one(plan, progress_every):
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"START MERGE {plan.channel}: {plan.output}", flush=True)
    count = 0
    with tifffile.TiffWriter(plan.partial, bigtiff=True) as writer:
        for part in plan.parts:
            with tifffile.TiffFile(part.path) as src:
                for page in src.pages:
                    writer.write(page.asarray(), contiguous=True,
                                 photometric="minisblack", metadata=None)
                    count += 1
                    if count % progress_every == 0 or count == plan.shape[0]:
                        print(f"{plan.channel}: copied {count}/{plan.shape[0]} planes", flush=True)
            print(f"{plan.channel}: finished {part.path.name}", flush=True)
    if count != plan.shape[0]:
        raise RuntimeError(f"{plan.channel}: expected {plan.shape[0]} planes, wrote {count}")
    validate_output(plan.partial, plan.shape, plan.dtype, label="TEMPORARY TIFF")
    os.link(plan.partial, plan.output)  # Atomic, no accidental overwrite.
    plan.partial.unlink()
    print(f"SUCCESS {plan.channel}: {plan.output}", flush=True)


def verify_pixels(plan, progress_every, mode, samples_per_task):
    validate_output(plan.output, plan.shape, plan.dtype)
    count = 0
    checked = 0
    if mode == "full":
        print(f"FULL PIXEL VERIFICATION {plan.channel}: checking every plane", flush=True)
    else:
        print(f"FAST VERIFICATION {plan.channel}: sampling up to "
              f"{samples_per_task} planes per task", flush=True)
    with tifffile.TiffFile(plan.output) as whole:
        for part in plan.parts:
            with tifffile.TiffFile(part.path) as source:
                n = len(source.pages)
                if mode == "full":
                    local_indices = range(n)
                else:
                    local_indices = sorted({
                        round(index * (n - 1) / (samples_per_task - 1))
                        for index in range(samples_per_task)
                    })
                for local_index in local_indices:
                    global_index = count + local_index
                    if global_index >= plan.shape[0] or not np.array_equal(
                            source.pages[local_index].asarray(),
                            whole.pages[global_index].asarray()):
                        raise ValueError(f"{plan.channel}: source differs at "
                                         f"output Z index {global_index}; NO deletion")
                    checked += 1
                    if mode == "full" and (checked % max(64, progress_every) == 0
                                            or checked == plan.shape[0]):
                        print(f"{plan.channel}: verified {checked}/{plan.shape[0]} planes", flush=True)
                count += n
    if count != plan.shape[0]:
        raise ValueError(f"{plan.channel}: only verified {count}/{plan.shape[0]} planes")
    print(f"VERIFICATION PASSED {plan.channel}: "
          f"{checked}/{plan.shape[0]} planes compared; "
          f"mode={mode}; cleanup permitted", flush=True)


def delete_sources(plan):
    # Only the explicit, task-level TIFFs in this validated plan are removed.
    # Never clean whole directories or files discovered with a new wildcard.
    paths = [part.path for part in plan.parts]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Source changed before deletion; aborting: {path}")
    recovered = sum(path.stat().st_size for path in paths) / 2**30
    for path in paths:
        path.unlink()
        print(f"DELETED VERIFIED TASK TIFF: {path}", flush=True)
    print(f"CLEANUP {plan.channel}: approximately {recovered:.2f} GiB of "
          "task TIFFs removed; full-brain TIFF and task metadata preserved", flush=True)


def main():
    args = arguments()
    run_dir = Path(args.run_dir).resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError(f"Not a run directory: {run_dir}")
    settings_path = run_dir / "leonardo_settings.json"
    if not settings_path.is_file():
        raise ValueError(f"Missing saved run settings: {settings_path}")
    run_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    suffix = run_settings.get("output", {}).get("suffix", "_destriped")
    if not isinstance(suffix, str) or re.fullmatch(r"[A-Za-z0-9_.-]*", suffix) is None:
        raise ValueError(f"Unsafe output.suffix in run settings: {suffix!r}")
    output_parent = (Path(args.output_parent) if args.output_parent
                     else (run_dir / "channels")).resolve()
    manifests = discover_manifests(run_dir)
    requested = args.channels or sorted(manifests)
    for name in requested:
        if name not in manifests:
            raise ValueError(f"Unknown channel {name!r}; available: {sorted(manifests)}")
    print(f"Run: {run_dir}", flush=True)
    print(f"Channels: {', '.join(requested)}", flush=True)
    print(f"Output filename suffix from JSON: {suffix!r}", flush=True)
    print(f"Verification: {args.verify}", flush=True)

    # Check every selected channel before writing even a temporary output.
    plans = [make_plan(name, manifests[name], args.match, output_parent, suffix)
             for name in requested]
    to_write = [plan for plan in plans if not plan.already_merged]
    if to_write:
        required = sum(plan.bytes_needed for plan in to_write) + int(args.reserve_gib * 2**30)
        # Find the nearest existing ancestor, without creating directories
        # during preflight. No fixed 600-GiB threshold.
        space_path = output_parent
        while not space_path.exists():
            space_path = space_path.parent
        free = shutil.disk_usage(space_path).free
        print(f"Free space: {free / 2**30:.2f} GiB; required including "
              f"{args.reserve_gib:g}-GiB reserve: {required / 2**30:.2f} GiB", flush=True)
        if free < required:
            raise RuntimeError("Insufficient free storage; no TIFFs created/deleted")
    if args.dry_run:
        print("DRY RUN PASSED: no files written or deleted", flush=True)
        return

    for plan in to_write:
        merge_one(plan, args.progress_every)

    # Always check output structure and source/output agreement. The default
    # samples evenly across every part; --verify full checks every pixel.
    for plan in plans:
        verify_pixels(plan, args.progress_every, args.verify, args.samples_per_task)

    if args.delete_sources_after_verify:
        # Verify every requested channel BEFORE deleting ANY selected source.
        for plan in plans:
            delete_sources(plan)
    else:
        print("All original task TIFFs preserved (cleanup not requested)", flush=True)
    print("ALL SELECTED CHANNELS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
PY
