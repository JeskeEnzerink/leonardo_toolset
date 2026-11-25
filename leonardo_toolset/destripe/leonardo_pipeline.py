#!/usr/bin/env python3
"""Support for Leonardo's sequential multichannel, four-GPU destriping pipeline.

Subcommands:
  plan prepare|chunk-config|validate|self-test
  postprocess --raw ... --output ... --config ...
  merge-task --task-dir ... --output ...
  merge-all RUN_DIR [--dry-run]

The underlying algorithms are unchanged from the four original helper modules.
Heavy scientific dependencies load only for subcommands that need them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from dataclasses import dataclass

def resolve_postprocessing_threshold(pp, zarr_path=None):
    import math

    value = pp.get("threshold", 200)

    if value == "auto":
        path = pp.get("intensity_json")
        if path is None and zarr_path is not None:
            # The stitching pipeline writes this sidecar next to each OME-Zarr.
            path = str(zarr_path) + ".intensity.json"

        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(
                "Automatic threshold requires an absolute intensity_json path"
            )

        stats = json.loads(Path(path).read_text())

        local_min = float(stats["local_min"])
        local_95 = float(stats["local_95"])

        if not all(map(math.isfinite, (local_min, local_95))):
            raise ValueError("Intensity JSON contains non-finite values")
        if local_min > local_95:
            raise ValueError("local_min cannot exceed local_95")

        threshold = (local_min + local_95) / 2

        print(
            f"AUTO threshold: ({local_min} + {local_95}) / 2 "
            f"= {threshold} | Source: {path}",
            flush=True
        )

    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid postprocessing threshold: {value!r}")

        threshold = float(value)

    if not math.isfinite(threshold):
        raise ValueError("Postprocessing threshold must be finite")

    return threshold

# -------------------------------------------------------
# Channel-aware, four-tasks-PER-channel Z planning and immutable chunk settings
# -------------------------------------------------------

GPU_TASKS = 4


def split_z(start, end, task, workers):
    if not (1 <= workers <= GPU_TASKS and 0 <= task < workers and 0 <= start <= end):
        raise ValueError('Invalid Z selection or array task index')
    q, r = divmod(end - start, workers)
    first = start + task * q + min(task, r)
    last = first + q + (task < r)
    return first, last


def split_four(start, end, task):
    """Original single-channel behavior; retained for older callers."""
    return split_z(start, end, task, GPU_TASKS)


def enabled_channels(cfg):
    """Select channels, discovering unique Zarr paths when not supplied explicitly."""
    inp = cfg['input']
    if 'channels' not in inp:
        # Backward compatible with existing single-Zarr settings.
        path = inp.get('zarr_path')
        if not isinstance(path, str) or not path.strip():
            raise ValueError('input.zarr_path must be set for a single-channel configuration')
        return [{'name': 'default', 'zarr_path': path, 'enabled': True}]
    channels = inp['channels']
    if not isinstance(channels, list) or not channels:
        raise ValueError('input.channels must be a nonempty list')
    names = set()
    selected = []
    for channel in channels:
        if not isinstance(channel, dict):
            raise ValueError('Each input.channels entry must be an object')
        name = channel.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid channel name {name!r}; use letters, digits, _ or -')
        if name in names:
            raise ValueError(f'Duplicate channel name: {name}')
        names.add(name)
        flag = channel.get('enabled')
        if not isinstance(flag, bool):
            raise ValueError(f'Channel {name}: enabled must be true or false')
        if flag:
            path = channel.get('zarr_path')
            if path is None or path == '':
                base_dir = inp.get('zarr_dir')
                dataset_id = inp.get('dataset_id')
                if not isinstance(base_dir, str) or not Path(base_dir).is_absolute():
                    raise ValueError(
                        f'Channel {name}: set an absolute input.zarr_dir '
                        'or specify the channel zarr_path explicitly'
                    )
                if not isinstance(dataset_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', dataset_id):
                    raise ValueError(f'Channel {name}: input.dataset_id must contain only letters, digits, _ or -')
                directory = Path(base_dir)
                if not directory.is_dir():
                    raise FileNotFoundError(f'Channel {name}: Zarr directory does not exist: {directory}')
                pattern = f'{dataset_id}_{name}_*.ome.zarr'
                matches = sorted(p for p in directory.glob(pattern) if p.is_dir())
                if len(matches) != 1:
                    found = ', '.join(p.name for p in matches) or '(none)'
                    raise ValueError(
                        f'Channel {name}: expected exactly one Zarr matching '
                        f'{directory / pattern}; found {len(matches)}: {found}. '
                        'Set channel.zarr_path explicitly if needed.'
                    )
                path = str(matches[0])
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f'Channel {name}: zarr_path must be a nonempty string')
            if not Path(path).is_absolute():
                raise ValueError(f'Channel {name}: zarr_path must be an absolute path')
            # Keep the original JSON unchanged; downstream consumers always see a path.
            selected.append({**channel, 'zarr_path': str(Path(path))})
    if not 1 <= len(selected) <= GPU_TASKS:
        raise ValueError(f'Enable between 1 and {GPU_TASKS} channels for the {GPU_TASKS}-GPU launcher')
    paths = [str(Path(c['zarr_path'])) for c in selected]
    if len(paths) != len(set(paths)):
        raise ValueError('Each enabled channel must refer to a different Zarr store')
    return selected


def channel_task_groups(cfg):
    """Four global task IDs per enabled channel, in channel order."""
    selected = enabled_channels(cfg)
    return [(channel, tuple(range(index * GPU_TASKS, (index + 1) * GPU_TASKS)))
            for index, channel in enumerate(selected)]


def task_channel(cfg, task):
    if not 0 <= task < GPU_TASKS * len(enabled_channels(cfg)):
        raise ValueError(f'Invalid GPU task ID: {task}')
    for channel, task_ids in channel_task_groups(cfg):
        if task in task_ids:
            return channel, task_ids.index(task), len(task_ids)
    raise AssertionError('Unassigned GPU task')


def numeric_range(env, prefix, dim, input_config):
    a_raw, b_raw = env.get(prefix + '_START', ''), env.get(prefix + '_END', '')
    # An explicit environment variable overrides leonardo_settings.json;
    # null means the full available extent.
    default_a = input_config.get(prefix.lower() + '_start')
    default_b = input_config.get(prefix.lower() + '_end')
    a = int(a_raw) if a_raw.strip() else (int(default_a) if default_a is not None else 0)
    b = int(b_raw) if b_raw.strip() else (int(default_b) if default_b is not None else dim)
    if not 0 <= a < b <= dim:
        raise ValueError(f'{prefix} range [{a}, {b}) invalid for dimension {dim}')
    return a, b


def stem_from_zarr(path):
    n = Path(path).name
    for suffix in ('.ome.zarr', '.zarr'):
        if n.endswith(suffix):
            return n[:-len(suffix)]
    return n


def write_immutable(path, data):
    p = Path(path)
    if p.exists():
        existing = json.loads(p.read_text())
        if existing != data:
            raise RuntimeError(f'Existing metadata/config differs; will not overwrite: {p}')
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')


def prepare(args):
    import tensorstore as ts
    source = Path(args.source).resolve()
    source_text = source.read_bytes()
    cfg = json.loads(source_text)
    if cfg.get('training', {}).get('backend') != 'jax':
        raise ValueError('This launcher requires JAX training in base configuration')
    if cfg.get('destripe', {}).get('angle_offset') is None:
        raise ValueError('angle_offset is missing from base config')
    channel, channel_task, channel_workers = task_channel(cfg, args.task)
    if 'channels' in cfg['input'] and os.environ.get('INPUT_ZARR'):
        raise ValueError('INPUT_ZARR override is unsafe with input.channels; edit channel zarr_path instead')
    zarr = Path(os.environ.get('INPUT_ZARR') or channel['zarr_path']).resolve()
    level = str(os.environ.get('ZARR_LEVEL') or cfg['input'].get('zarr_level', 0))
    store = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file', 'path': str(zarr / level)}}, read=True).result()
    shape = tuple(map(int, store.shape))
    if len(shape) != 3:
        raise ValueError(f'Expected a 3D ZYX Zarr level, got shape={shape}')
    z0, z1 = numeric_range(os.environ, 'Z', shape[0], cfg['input'])
    y0, y1 = numeric_range(os.environ, 'Y', shape[1], cfg['input'])
    x0, x1 = numeric_range(os.environ, 'X', shape[2], cfg['input'])
    a, b = split_z(z0, z1, channel_task, channel_workers)
    m = {
        'schema': 2, 'task': args.task, 'precision': args.precision,
        'channel_name': channel['name'], 'channel_task': channel_task,
        'channel_workers': channel_workers,
        'source_config': str(source), 'source_config_sha256': hashlib.sha256(source_text).hexdigest(),
        'zarr_path': str(zarr), 'zarr_level': level, 'zarr_shape': list(shape),
        'source_stem': stem_from_zarr(zarr),
        'requested_z': [z0, z1], 'task_z': [a, b],
        'y_range': [y0, y1], 'x_range': [x0, x1],
        'shape_yx': [y1 - y0, x1 - x0],
        'block_size': 8, 'n_blocks': (b - a + 7) // 8,
        'epochs': 150, 'warm_start': False, 'fft_alpha': 1., 'fft_width': 5.,
    }
    write_immutable(Path(args.task_dir) / 'manifest.json', m)
    # Printed fields each on separate lines (filenames may contain spaces).
    for item in [a, b, y1-y0, x1-x0, m['n_blocks'], m['source_stem']]:
        print(item)


def chunk_config(args):
    manifest = json.loads(Path(args.manifest).read_text())
    src = Path(manifest['source_config'])
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['source_config_sha256']:
        raise RuntimeError('Base config has changed since this run was planned!')
    cfg = json.loads(raw)
    channels = cfg["input"].get("channels", [])

    channel_settings = next(
        (c for c in channels if c["name"] == manifest["channel_name"]),
        {}
    )

    cfg["postprocessing"] = channel_settings.get(
        "postprocessing",
        {"method": "original_fft"}
    )
    a, b = args.start, args.end
    lo, hi = manifest['task_z']
    if not (lo <= a < b <= hi and b - a <= manifest['block_size']):
        raise ValueError('Chunk range lies outside assigned task or exceeds 8 planes')
    y0, y1 = manifest['y_range']
    x0, x1 = manifest['x_range']
    cfg['input'].update(zarr_path=manifest['zarr_path'], zarr_level=manifest['zarr_level'],
                        z_start=a, z_end=b, y_start=y0, y_end=y1, x_start=x0, x_end=x1)
    cfg['input'].pop('channels', None)  # Trainer consumes one Zarr per chunk.
    cfg['input'].pop('zarr_dir', None)  # Discovery is only used by the launcher.
    cfg['input'].pop('dataset_id', None)
    cfg['training']['backend'] = 'jax'
    cfg['training']['n_epochs'] = 150
    cfg['warm_start'] = {'enabled': False, 'epochs_after_first': 150}
    cfg.setdefault('analysis', {})
    cfg['analysis'] = {'plot_epoch_loss': False, 'save_epoch_loss_csv': False}
    cfg['output']['save_dir'] = str(Path(args.chunks).resolve())
    cfg['output']['suffix'] = (f"_dyn_{manifest['precision']}_t{manifest['task']}_"
                               f"z{a:04d}_{b-1:04d}")
    write_immutable(args.config, cfg)
    expected_raw = Path(args.chunks).resolve() / (manifest['source_stem'] + cfg['output']['suffix'] + '.tif')
    print(expected_raw)


def plan_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    prep = s.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--task', required=True, type=int)
    prep.add_argument('--task-dir', required=True)
    prep.add_argument('--precision', choices=('fp64', 'fp32', 'fp16'), required=True)
    conf = s.add_parser('chunk-config')
    conf.add_argument('--manifest', required=True)
    conf.add_argument('--start', type=int, required=True)
    conf.add_argument('--end', type=int, required=True)
    conf.add_argument('--config', required=True)
    conf.add_argument('--chunks', required=True)
    validate = s.add_parser('validate')
    validate.add_argument('--source', required=True)
    count = s.add_parser('count')
    count.add_argument('--source', required=True)
    t = s.add_parser('self-test')
    a = p.parse_args(argv)
    if a.command == 'prepare':
        prepare(a)
    elif a.command == 'chunk-config':
        chunk_config(a)
    elif a.command == 'validate':
        cfg = json.loads(Path(a.source).read_text())
        for channel, ids in channel_task_groups(cfg):
            print(f"Channel {channel['name']}: task(s) {', '.join(map(str, ids))} -> {channel['zarr_path']}")
            pp = channel.get('postprocessing', {})
            if pp.get('threshold') == 'auto':
                resolved_threshold = resolve_postprocessing_threshold(pp, channel['zarr_path'])
                print(f"Channel {channel['name']}: auto postprocessing threshold = {resolved_threshold}")
        print('PASS: four GPU tasks per channel, no shared task IDs.')
    elif a.command == 'count':
        cfg = json.loads(Path(a.source).read_text())
        print(len(channel_task_groups(cfg)))
    else:
        for n_channels in range(1, GPU_TASKS + 1):
            cfg = {'input': {'channels': [
                {'name': f'C{i}', 'enabled': True, 'zarr_path': f'/data/C{i}.ome.zarr'}
                for i in range(n_channels)]}}
            groups = channel_task_groups(cfg)
            assert [task for _, ids in groups for task in ids] == list(range(GPU_TASKS * n_channels))
            for z in range(1, 18):
                for channel, ids in groups:
                    pairs = [split_z(23, 23 + z, i, len(ids)) for i in range(len(ids))]
                    assert pairs[0][0] == 23 and pairs[-1][1] == 23 + z
                    assert all(pairs[i][1] == pairs[i+1][0] for i in range(len(ids)-1))
                    assert max(b-a for a,b in pairs) - min(b-a for a,b in pairs) <= 1
                    for local_task, global_task in enumerate(ids):
                        got, index, total = task_channel(cfg, global_task)
                        assert (got['name'], index, total) == (channel['name'], local_task, len(ids))
        print('PASS: 1..4 channels, four GPU tasks per channel, all 1..17-plane Z ranges covered exactly once.')


# -------------------------------------------------------
# Original-referenced alpha=1.0, FFT-width=5 postprocessing
# -------------------------------------------------------

def check_tiff(path, n, shape):
    import numpy as np
    import tifffile
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing or empty TIFF: {path}')
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != n:
            raise RuntimeError(f'{path}: {len(tif.pages)} pages; expected {n}')
        for i, page in enumerate(tif.pages):
            if tuple(page.shape) != tuple(shape) or np.dtype(page.dtype) != np.dtype('uint16'):
                raise RuntimeError(f'{path}: page {i} is {page.shape}, {page.dtype}; '
                                   f'expected {shape}, uint16')
    return path


def postprocess_main(argv=None):
    import numpy as np
    import tifffile
    import tensorstore as ts
    from leonardo_alpha1_full256 import filter_plane
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    inp = cfg['input']
    start, end = int(inp['z_start']), int(inp['z_end'])
    y0, y1 = int(inp['y_start']), int(inp['y_end'])
    x0, x1 = int(inp['x_start']), int(inp['x_end'])
    shape = (y1-y0, x1-x0)
    if not (1 <= end-start <= 8 and all(v > 0 for v in shape)):
        raise ValueError('Invalid chunk size or ROI dimensions')
    raw = check_tiff(a.raw, end-start, shape)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    tmp = out.with_name(out.name + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Old incomplete output exists: {tmp}')
    store = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file',
        'path': str(Path(inp['zarr_path']) / str(inp['zarr_level']))}}, read=True).result()
    angles = cfg['destripe']['angle_offset']
    vertical = cfg['destripe']['is_vertical']
    pp = cfg.get("postprocessing", {})
    method = pp.get("method", "original_fft")
    threshold = resolve_postprocessing_threshold(pp, inp['zarr_path'])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tifffile.TiffFile(raw) as src, tifffile.TiffWriter(tmp, bigtiff=True) as dest:
            for i, z in enumerate(range(start, end)):
                original = np.asarray(store[z, y0:y1, x0:x1].read().result())
                leonardo = src.pages[i].asarray()
                if original.shape != shape or leonardo.shape != shape:
                    raise RuntimeError(f'Z{z}: input/raw ROI mismatch: {original.shape}, {leonardo.shape}, {shape}')


                if method == "smooth_weighted":

                    from leonardo_interpolation_pilot import smooth_postprocess

                    try:
                        result, _ = smooth_postprocess(
                            original.astype(np.float32),
                            leonardo.astype(np.float32),
                            cfg,
                            threshold=threshold,
                            margin=int(pp.get("margin", 24)),
                            sigma=float(pp.get("sigma", 48)),
                            floor=float(pp.get("support_floor", 0.03))
                        )

                    except ValueError as exc:
                        if str(exc) != "No valid tissue interior; cannot test interpolation":
                            raise

                        print(
                            f"Z{z}: no valid tissue for smoothing; preserving original",
                            flush=True
                        )
                        result = original.astype(np.uint16)

                elif method == "original_fft":

                    result = filter_plane(
                        original,
                        leonardo,
                        angles,
                        vertical=vertical,
                        angular_width=5.,
                        period_px=192.,
                        pad=256,
                        threshold=threshold,
                        margin=int(pp.get("margin", 24))
                    )

                else:
                    raise ValueError(f"Unknown postprocessing method: {method}")
                # Minor FFT overshoot around the uint16 limits should not
                # invalidate hours of otherwise successful GPU training.
                # Check AFTER rounding: e.g. 65535.2 rounds to 65535 and
                # therefore would not be clipped in the stored image.
                if not np.isfinite(result).all():
                    raise RuntimeError(f'Z{z}: non-finite FFT postprocessing values')
                rounded = np.rint(result)
                clipped_lo = int(np.count_nonzero(rounded < 0))
                clipped_hi = int(np.count_nonzero(rounded > 65535))
                clipped_total = clipped_lo + clipped_hi
                # At most 0.001% of pixels may saturate; widespread clipping
                # still triggers an error instead of silently damaging data.
                clip_limit = max(1, int(rounded.size * 0.00001))
                if clipped_total > clip_limit:
                    raise RuntimeError(
                        f'Z{z}: excessive uint16 clipping: {clipped_lo} low, '
                        f'{clipped_hi} high of {rounded.size} pixels '
                        f'(limit {clip_limit}); unrounded min/max '
                        f'{float(result.min()):.3f}/{float(result.max()):.3f}'
                    )
                if clipped_total:
                    print(
                        f'WARNING: Z{z}: saturating {clipped_lo} low and '
                        f'{clipped_hi} high pixels out of {rounded.size}; '
                        f'original min/max={float(result.min()):.3f}/'
                        f'{float(result.max()):.3f}', flush=True,
                    )
                dest.write(np.clip(rounded,0,65535).astype(np.uint16),
                           contiguous=True, photometric='minisblack', metadata=None)
                print(f'FFT alpha1 width5 complete Z{z}; clipped={clipped_total}', flush=True)
        check_tiff(tmp, end-start, shape)
        os.replace(tmp, out)
        print(f'VALIDATED CORRECTED CHUNK: {out}', flush=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# -------------------------------------------------------
# Validated per-task merge of eight-plane corrected chunks
# -------------------------------------------------------

def merge_task_main(argv=None):
    import tifffile
    p = argparse.ArgumentParser()
    p.add_argument('--task-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    root = Path(a.task_dir)
    meta = json.loads((root/'manifest.json').read_text())
    first, last = meta['task_z']
    n = last - first
    if n < 1:
        print('Empty task, nothing to merge')
        return
    h, w = meta['shape_yx']
    chunks = []
    for b in range(meta['n_blocks']):
        start = first + b*meta['block_size']
        end = min(start + meta['block_size'], last)
        chunk = root/'chunks'/f'alpha1w5_z{start:04d}_{end-1:04d}.tif'
        check_tiff(chunk, end-start, (h,w))
        chunks.append(chunk)
    out = Path(a.output)
    if out.exists():
        check_tiff(out,n,(h,w))
        print(f'FINAL ALREADY VALIDATED: {out}',flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    required = n*h*w*2
    free = shutil.disk_usage(out.parent).free
    if free < required + 2_000_000_000:
        raise RuntimeError(f'Insufficient filesystem space to merge: free {free/1e9:.1f}GB, '
                           f'need at least {(required+2_000_000_000)/1e9:.1f}GB. Corrected chunks kept.')
    partial = out.with_name(out.name+'.partial')
    if partial.exists():
        raise FileExistsError(f'Existing incomplete merge: {partial}; inspect before retrying')
    try:
        with tifffile.TiffWriter(partial,bigtiff=True) as dest:
            for i,chunk in enumerate(chunks,1):
                with tifffile.TiffFile(chunk) as src:
                    for page in src.pages:
                        dest.write(page.asarray(), contiguous=True,
                                   photometric='minisblack', metadata=None)
                print(f'MERGED {i}/{len(chunks)}: {chunk.name}',flush=True)
        check_tiff(partial,n,(h,w))
        os.replace(partial,out)
        print(f'FINAL VALIDATED: {out} shape=({n},{h},{w}) uint16',flush=True)
        for chunk in chunks:
            chunk.unlink()
        print('Deleted ONLY merged corrected TIFF chunks; input data and final TIFF preserved.',flush=True)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


# -------------------------------------------------------
# Validated full-volume merge of four task TIFFs
# -------------------------------------------------------

FILENAME = re.compile(
    r"^(?P<stem>.+)_z(?P<start>\d+)_(?P<end>\d+)_(?P<variant>.+)\.tif$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Part:
    path: Path
    start: int
    end: int
    stem: str
    variant: str
    width: int
    shape: tuple[int, int, int]
    dtype: np.dtype


def task_root_for_run(run_dir: Path) -> Path:
    """Support both existing run/task_0 and new run/tasks/task_0 layouts."""
    nested = run_dir / 'tasks'
    legacy_exists = (run_dir / 'task_0').is_dir()
    nested_exists = (nested / 'task_0').is_dir()
    if legacy_exists and nested_exists:
        raise ValueError(f'Ambiguous task layout: both {run_dir} and {nested} contain task_0')
    if legacy_exists:
        return run_dir
    return nested if nested.is_dir() else run_dir


def merged_output_suffix(cfg: dict) -> str:
    """Use the run's saved JSON suffix, e.g. '_destriped'."""
    suffix = cfg.get('output', {}).get('suffix', '_destriped')
    if not isinstance(suffix, str) or re.fullmatch(r'[A-Za-z0-9_.-]*', suffix) is None:
        raise ValueError(f'Unsafe output.suffix in run settings: {suffix!r}')
    return suffix


def select_merged_output(parent: Path, parts: list[Part], suffix: str) -> Path:
    """Use stem + JSON suffix, without duplicating previously merged TIFFs."""
    first, last = parts[0], parts[-1]
    width = max(part.width for part in parts)
    candidates = (
        parent / f'{first.stem}{suffix}.tif',
        parent / f'{first.stem}.tif',
        parent / f'{first.stem}_fullbrain.tif',
        parent / f'{first.stem}_fullbrain_z{first.start:0{width}d}_{last.end:0{width}d}.tif',
        parent / (f'{first.stem}_fullbrain_z{first.start:0{width}d}_'
                  f'{last.end:0{width}d}_{first.variant}.tif'),
    )
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if len(existing) > 1:
        raise ValueError(f'Multiple possible merged TIFFs exist; inspect before merging: {existing}')
    return existing[0] if existing else candidates[0]


def discover_parts(run_dir: Path, pattern: str, expected_parts: int,
                   task_indices: tuple[int, ...] | None = None) -> list[Part]:
    import numpy as np
    import tifffile
    if task_indices is None:
        task_indices = tuple(range(expected_parts))
    if len(task_indices) != expected_parts or len(task_indices) != len(set(task_indices)):
        raise ValueError('Task ID list must contain exactly the expected number of unique tasks')
    task_root = task_root_for_run(run_dir)
    task_dirs = sorted(
        (p for p in task_root.iterdir() if p.is_dir() and re.fullmatch(r"task_\d+", p.name)),
        key=lambda p: int(p.name.split("_")[1]),
    )
    cfg = json.loads((run_dir / 'leonardo_settings.json').read_text())
    total_tasks = GPU_TASKS * len(enabled_channels(cfg))
    if [int(p.name.split('_')[1]) for p in task_dirs] != list(range(total_tasks)):
        raise ValueError(
            f"Expected task_0 through task_{total_tasks-1} in {task_root}"
        )
    task_dirs = [task_root / f'task_{i}' for i in task_indices]

    parts: list[Part] = []
    shared_identity = None
    expected_z_range = None
    for task_dir in task_dirs:
        manifest_path = task_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'Missing run manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        task_index = int(task_dir.name.split('_')[1])
        if manifest['task'] != task_index or not 0 <= task_index < total_tasks:
            raise ValueError(f'Manifest task number mismatch: {manifest_path}')
        identity = (manifest['source_config_sha256'], manifest['zarr_path'],
                    str(manifest['zarr_level']), tuple(manifest['requested_z']),
                    tuple(manifest['y_range']), tuple(manifest['x_range']),
                    tuple(manifest['shape_yx']), manifest['precision'],
                    manifest.get('channel_name', 'default'))
        if shared_identity is None:
            shared_identity = identity
            expected_z_range = tuple(manifest['requested_z'])
        elif identity != shared_identity:
            raise ValueError(f'Incompatible task manifest: {manifest_path}')
        expected_start, expected_end = manifest['task_z']
        matches = sorted(p for p in task_dir.glob(pattern) if p.is_file())
        if expected_start == expected_end:
            if matches:
                raise ValueError(f'Empty task should not have a TIFF: {task_dir}')
            print(f'EMPTY TASK {task_index}: no input TIFF needed', flush=True)
            continue
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one finished TIFF matching {pattern!r} in "
                f"{task_dir}, found {len(matches)}: {[p.name for p in matches]}"
            )

        path = matches[0]
        m = FILENAME.fullmatch(path.name)
        if m is None:
            raise ValueError(f"Cannot read Z range from TIFF filename: {path.name}")
        start, end = int(m["start"]), int(m["end"])
        if (start, end + 1) != (expected_start, expected_end):
            raise ValueError(f'{path.name} does not match manifest range {expected_start}:{expected_end}')
        if start > end:
            raise ValueError(f"Invalid Z range in {path.name}: {start}–{end}")

        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            shape = tuple(int(x) for x in series.shape)
            dtype = np.dtype(series.dtype)
            page_count = len(tf.pages)
            if len(shape) == 2 and page_count == 1:
                shape = (1, *shape)
        if len(shape) != 3 or shape[0] != end - start + 1 or page_count != shape[0]:
            raise ValueError(
                f"Unexpected Z shape/pages for {path.name}: shape={shape}, "
                f"pages={page_count}, expected Z={end - start + 1}"
            )
        if dtype != np.dtype("uint16"):
            raise ValueError(f"Expected uint16 output, got {dtype} in {path.name}")
        if tuple(shape[1:]) != tuple(manifest['shape_yx']):
            raise ValueError(f'XY dimensions disagree with manifest: {path.name}')

        parts.append(
            Part(
                path=path,
                start=start,
                end=end,
                stem=m["stem"],
                variant=m["variant"],
                width=max(4, len(m["start"]), len(m["end"])),
                shape=shape,
                dtype=dtype,
            )
        )

    if not parts:
        raise ValueError('No output TIFFs available')
    parts.sort(key=lambda x: x.start)
    first = parts[0]
    if (first.start, parts[-1].end + 1) != expected_z_range:
        raise ValueError(f'Merged range would not cover requested {expected_z_range}')
    for previous, current in zip(parts, parts[1:]):
        if current.start != previous.end + 1:
            raise ValueError(
                f"Gap or overlap: {previous.path.name} ends at Z{previous.end}, "
                f"but {current.path.name} starts at Z{current.start}"
            )
    for part in parts:
        if part.shape[1:] != first.shape[1:]:
            raise ValueError(f"XY mismatch: {part.path.name}: {part.shape[1:]} vs {first.shape[1:]}")
        if part.stem != first.stem or part.variant != first.variant or part.dtype != first.dtype:
            raise ValueError(f"Incompatible series name, processing variant, or dtype: {part.path.name}")
    return parts


def merge(run_dir: Path, pattern: str, expected_parts: int, output: Path | None,
          dry_run: bool, task_indices: tuple[int, ...] | None = None) -> None:
    import numpy as np
    import tifffile
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    parts = discover_parts(run_dir, pattern, expected_parts, task_indices)
    first, last = parts[0], parts[-1]
    if output is None:
        cfg = json.loads((run_dir / 'leonardo_settings.json').read_text())
        output = select_merged_output(run_dir, parts, merged_output_suffix(cfg))
    output = output.resolve()
    temporary = output.with_name(output.stem + ".incomplete.tif")

    z_count = last.end - first.start + 1
    shape = (z_count, *first.shape[1:])
    needed = int(np.prod(shape, dtype=np.int64) * first.dtype.itemsize)

    print("Validated inputs in Z order:", flush=True)
    for part in parts:
        print(f"  Z{part.start}–{part.end}: {part.path} shape={part.shape}", flush=True)
    print(f"Output: {output}\nShape: {shape}, dtype={first.dtype}", flush=True)
    print(f"Estimated raw size: {needed / 2**30:.2f} GiB", flush=True)

    if output.exists():
        with tifffile.TiffFile(output) as tf:
            got = tuple(int(x) for x in tf.series[0].shape)
            if len(got) == 2 and z_count == 1:
                got = (1, *got)
            got_dtype = np.dtype(tf.series[0].dtype)
            got_pages = len(tf.pages)
        if got != shape or got_dtype != first.dtype or got_pages != z_count:
            raise SystemExit(f'EXISTING OUTPUT IS INVALID; will not overwrite: {output}')
        print(f'FINAL OUTPUT ALREADY VALIDATED: {output}', flush=True)
        return
    if temporary.exists():
        raise SystemExit(f"INCOMPLETE OUTPUT EXISTS; inspect before removing: {temporary}")
    if not output.parent.is_dir() and not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir() and not dry_run:
        raise SystemExit(f"Output directory does not exist: {output.parent}")

    if dry_run:
        print("DRY RUN PASSED: no files created", flush=True)
        return
    free = shutil.disk_usage(output.parent).free
    print(f"Free space: {free / 2**30:.2f} GiB", flush=True)
    if free < needed + 5 * 2**30:
        raise SystemExit("Not enough free space for merged TIFF plus 5 GiB safety margin")

    # This creates a BigTIFF on disk without loading the full brain into RAM.
    arr = tifffile.memmap(
        temporary,
        shape=shape,
        dtype=first.dtype,
        bigtiff=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )
    offset = 0
    for part in parts:
        with tifffile.TiffFile(part.path) as tf:
            for page in tf.pages:
                arr[offset] = page.asarray()
                offset += 1
                if offset % 32 == 0:
                    arr.flush()
                    print(f"Copied {offset}/{z_count} planes", flush=True)
        arr.flush()
    del arr

    with tifffile.TiffFile(temporary) as tf:
        result = tf.series[0]
        if result.shape != shape or np.dtype(result.dtype) != first.dtype or len(tf.pages) != z_count:
            raise RuntimeError(
                f"Output validation failed: shape={result.shape}, "
                f"dtype={result.dtype}, pages={len(tf.pages)}"
            )

    # Hard-linking creates the finished name atomically and NEVER overwrites an
    # existing file, unlike os.replace(). The temporary name is then removed.
    os.link(temporary, output)
    temporary.unlink()
    print(f"SUCCESS: {output}", flush=True)
    print("All source TIFFs are preserved.", flush=True)


def merge_all_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run folder containing task_0 etc. directly or inside tasks/")
    parser.add_argument(
        "--match", default="*_alpha1_fftwidth5_uint16.tif",
        help="Pattern for one final TIFF directly inside each task directory",
    )
    parser.add_argument("--expected-parts", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional output TIFF (default: in run_dir)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing anything")
    args = parser.parse_args(argv)
    if args.expected_parts < 1:
        parser.error("--expected-parts must be at least 1")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f'Run directory does not exist: {run_dir}')
    settings_path = run_dir / 'leonardo_settings.json'
    if not settings_path.is_file():
        raise SystemExit(f'Missing run settings snapshot: {settings_path}')
    cfg = json.loads(settings_path.read_text())
    groups = channel_task_groups(cfg)
    if args.expected_parts != GPU_TASKS:
        raise ValueError(f'The launcher uses exactly {GPU_TASKS} GPU array tasks')
    if args.output is not None and len(groups) != 1:
        parser.error('--output is only valid when exactly one channel is enabled')

    # Validate the entire planned allocation, including empty tasks, before
    # writing any channel's merged TIFF.
    planned = []
    task_root = task_root_for_run(run_dir)
    for channel, ids in groups:
        for local_task, task_index in enumerate(ids):
            meta_path = task_root / f'task_{task_index}' / 'manifest.json'
            if not meta_path.is_file():
                raise ValueError(f'Missing run manifest: {meta_path}')
            meta = json.loads(meta_path.read_text())
            if meta['task'] != task_index or meta.get('channel_name', 'default') != channel['name']:
                raise ValueError(f'Task {task_index} is not assigned to channel {channel["name"]}')
            if meta.get('channel_task', local_task) != local_task or meta.get('channel_workers', len(ids)) != len(ids):
                raise ValueError(f'Wrong channel worker allocation in: {meta_path}')
            if str(Path(meta['zarr_path'])) != str(Path(channel['zarr_path']).resolve()):
                raise ValueError(f'Wrong Zarr source for {channel["name"]}: {meta_path}')
        destination = args.output
        if destination is None and 'channels' in cfg['input']:
            # Isolate channel products even when different inputs have the
            # same basename and therefore the same TIFF stem.
            parts = discover_parts(run_dir, args.match, len(ids), ids)
            destination = select_merged_output(run_dir / 'channels' / channel['name'],
                                               parts, merged_output_suffix(cfg))
        planned.append((channel['name'], ids, destination))

    print(f'Validating {len(groups)} enabled channel(s), {GPU_TASKS} GPU tasks each.', flush=True)
    for name, ids, output in planned:
        print(f'CHANNEL {name} (GPU tasks {ids})', flush=True)
        merge(run_dir, args.match, len(ids), output, dry_run=True, task_indices=ids)
    if args.dry_run:
        print('ALL CHANNELS VALIDATED: no files created', flush=True)
        return
    for name, ids, output in planned:
        print(f'MERGING CHANNEL {name}', flush=True)
        merge(run_dir, args.match, len(ids), output, dry_run=False, task_indices=ids)
    print(f'ALL {len(groups)} CHANNEL MERGES COMPLETE: source TIFFs preserved', flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python leonardo_pipeline.py '
            '{plan|postprocess|merge-task|merge-all} ...'
        )
    command, *arguments = sys.argv[1:]
    commands = {
        'plan': plan_main,
        'postprocess': postprocess_main,
        'merge-task': merge_task_main,
        'merge-all': merge_all_main,
    }
    if command not in commands:
        raise SystemExit(f'Unknown pipeline command: {command}')
    commands[command](arguments)


if __name__ == '__main__':
    main()




