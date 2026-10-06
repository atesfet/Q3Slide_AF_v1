"""Small, uncorrected mosaics for human review of candidate blank patches."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import traceback
import uuid

import numpy as np
from PIL import Image, ImageDraw

from pipeline_scripts.oct_slide_artifact import read_geometry, estimate_background_patches
from .pipeline import UserFacingError, inspect_geometry, validate_reference_patches


def display_image(array: np.ndarray) -> Image.Image:
    low, high = np.percentile(array, [1, 99.5])
    scaled = np.clip((array - low) / max(float(high-low), 1.0), 0, 1)
    return Image.fromarray(np.rint(np.sqrt(scaled) * 255).astype(np.uint8))


def save_reference_overlay(source: Path, destination: Path, rows: int, columns: int, selected):
    image = Image.open(source).convert('RGB')
    draw = ImageDraw.Draw(image)
    selected = set(tuple(p) for p in selected)
    for row in range(rows):
        for column in range(columns):
            x0, y0 = column*image.width//columns, row*image.height//rows
            x1, y1 = (column+1)*image.width//columns-1, (row+1)*image.height//rows-1
            chosen = (row, column) in selected
            draw.rectangle((x0, y0, x1, y1), outline='#77f3bc' if chosen else '#707b80', width=2 if chosen else 1)
            text = f'{row},{column}' + (' *' if chosen else '')
            bounds = draw.textbbox((x0+3, y0+2), text)
            draw.rectangle(bounds, fill='#102a28')
            draw.text((x0+3, y0+2), text, fill='#c5ffe0' if chosen else '#fff')
    image.save(destination)


def build_preview(config: dict, output: Path, update=lambda progress: None) -> dict:
    path = Path(config['image_path']).expanduser().resolve()
    fov = float(config['fov_um'])
    auto = bool(config.get('auto_mode', True))
    rows = None if auto or not config.get('rows') else int(config['rows'])
    columns = None if auto or not config.get('columns') else int(config['columns'])
    summary = inspect_geometry(path, fov, rows, columns)
    geometry = read_geometry(path, summary.rows, summary.columns, fov)
    if summary.rows * summary.columns < 2:
        raise UserFacingError('At least two acquisition patches are needed to learn a blank reference.')
    import tifffile
    try:
        volume = tifffile.memmap(path)
    except ValueError as exc:
        raise UserFacingError('This TIFF cannot be memory-mapped. Re-export the Z stack as an uncompressed TIFF, then try again.') from exc
    py, px = geometry.patch_shape_yx
    # Keep each tile equally sampled and keep the entire Z range. No Z crop or
    # correction is applied. Memory scales with the preview, not the raw mosaic.
    nz = geometry.shape_zyx[0]
    budget_side = int(np.sqrt(128 * 1024**2 / (nz * summary.rows * summary.columns * volume.dtype.itemsize)))
    if budget_side < 4:
        raise UserFacingError('This volume has too many slices or tiles for the reference preview. Supply known blank coordinates in manual mode.')
    side_y, side_x = min(64, py, budget_side), min(64, px, budget_side)
    ys = np.linspace(0, py-1, side_y, dtype=int)
    xs = np.linspace(0, px-1, side_x, dtype=int)
    sampled = np.empty((nz, summary.rows * side_y, summary.columns * side_x), dtype=volume.dtype)
    raw = np.empty(sampled.shape[1:], dtype=np.float32)
    contrast = np.empty_like(raw)
    for row in range(summary.rows):
        for column in range(summary.columns):
            tile = np.asarray(volume[:, row*py+ys[:, None], column*px+xs[None, :]])
            region = np.s_[row*side_y:(row+1)*side_y, column*side_x:(column+1)*side_x]
            sampled[:, region[0], region[1]] = tile
            values = tile.astype(np.float32)
            raw[region] = np.max(values, axis=0)
            excess = np.maximum(values - np.quantile(values, .2, axis=0)[None], 0)
            profile = np.median(excess, axis=(1, 2))
            contrast[region] = np.max(np.maximum(excess-profile[:, None, None], 0), axis=0)
            update(5 + int(70 * (row*summary.columns+column+1)/(summary.rows*summary.columns)))
    sample_geometry = replace(geometry, shape_zyx=sampled.shape, patch_shape_yx=(side_y, side_x))
    selected, report = estimate_background_patches(sampled, sample_geometry, count=min(8, summary.rows*summary.columns))
    output.mkdir(parents=True, exist_ok=True)
    display_image(raw).save(output/'raw_projection.png')
    display_image(contrast).save(output/'contrast_projection.png')
    result = {
        'image_path': str(path), 'file_mtime_ns': path.stat().st_mtime_ns,
        'geometry': summary.as_dict(), 'selected': [list(p) for p in selected], 'patches': report,
        'preview_shape_yx': list(raw.shape),
        'selection_basis': 'Provisional blank ranking from uniformly sampled raw tiles: low deep-tail signal and low variance. Review visually before use.',
        'projection_method': 'Raw maximum across the entire Z stack, sampled at up to 64 × 64 XY positions per tile. Contrast view removes each tile’s median depth profile for display only.',
    }
    (output/'preview_summary.json').write_text(json.dumps(result, indent=2))
    update(100)
    return result


class PreviewManager:
    def __init__(self, root: Path):
        self.root = root
        self.tasks = {}
        self.lock = threading.RLock()

    def busy(self):
        with self.lock:
            return any(t['status'] == 'running' for t in self.tasks.values())

    def create(self, config: dict):
        with self.lock:
            if self.busy():
                raise UserFacingError('A mosaic preview is already being prepared. Wait for it to finish.')
            task_id = uuid.uuid4().hex[:12]
            task = {'id': task_id, 'status': 'running', 'progress': 0, 'result': None, 'error': None}
            self.tasks[task_id] = task
        def work():
            def update(progress):
                with self.lock:
                    task['progress'] = progress
            try:
                result = build_preview(config, self.root/task_id, update)
                with self.lock:
                    task.update(status='completed', result=result)
            except Exception as exc:
                with self.lock:
                    task.update(status='failed', error=str(exc))
                traceback.print_exc()
        threading.Thread(target=work, daemon=True).start()
        return task.copy()

    def get(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            return task.copy() if task else None

    def validate_selection(self, config):
        task = self.get(str(config.get('reference_preview_id', '')))
        if not task or task['status'] != 'completed':
            raise UserFacingError('Generate a new reference preview before using reviewed tiles.')
        result = task['result']
        path = Path(config['image_path']).expanduser().resolve()
        if str(path) != result['image_path'] or path.stat().st_mtime_ns != result['file_mtime_ns']:
            raise UserFacingError('The input image changed. Generate a fresh reference preview.')
        auto = bool(config.get('auto_mode', True))
        geometry = inspect_geometry(path, float(config['fov_um']),
            None if auto or not config.get('rows') else int(config['rows']),
            None if auto or not config.get('columns') else int(config['columns']))
        if geometry.as_dict() != result['geometry']:
            raise UserFacingError('The FOV or grid changed. Generate a new reference preview.')
        validate_reference_patches(config.get('reference_patches'), geometry)
        return result
