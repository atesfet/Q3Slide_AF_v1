"""Lossless, traceable normalization of explicitly calibrated TIFF layouts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading

import numpy as np
import tifffile

_LOCK = threading.Lock()


def read_reconstruction_sidecar(path: Path, shape: tuple) -> dict | None:
    """Validate axis coordinates, not the generic TIFF page label or axis size."""
    from .pipeline import UserFacingError
    sidecar = Path(str(path)+'.json')
    if not sidecar.is_file():
        return None
    try:
        metadata = json.loads(sidecar.read_text())['metadata']
        lengths, spacing = {}, {}
        for axis in 'zyx':
            record = metadata[axis]
            coordinates = np.asarray(record['values'], dtype=float)
            units = str(record['units']).strip().lower()
            scale = {'mm': 1000., 'micron': 1., 'microns': 1., 'um': 1., 'µm': 1.}.get(units)
            if scale is None or coordinates.ndim != 1 or len(coordinates) < 2 or not np.all(np.isfinite(coordinates)):
                raise ValueError(f'Invalid {axis.upper()} coordinates or units')
            deltas = np.diff(coordinates)
            delta = float(np.median(deltas))
            if delta <= 0 or not np.allclose(deltas, delta, rtol=1e-5, atol=abs(delta)*1e-6):
                raise ValueError(f'{axis.upper()} coordinates must be uniformly increasing')
            lengths[axis.upper()] = len(coordinates)
            spacing[axis] = delta*scale
        candidates = [layout for layout in ('ZYX', 'YZX') if tuple(lengths[a] for a in layout) == tuple(shape)]
        if len(candidates) != 1:
            # Known reconstruction summary explicitly declares its storage order.
            summary_path = path.with_name(path.stem+'_run_summary.json')
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            if summary.get('output_shape_yzx') == list(shape) and tuple(lengths[a] for a in 'YZX') == tuple(shape):
                candidates = ['YZX']
            else:
                raise ValueError('Sidecar coordinates do not unambiguously match the TIFF shape')
        stat = sidecar.stat()
        return {'layout': candidates[0], 'voxels': tuple(spacing[a] for a in 'zyx'),
                'path': str(sidecar), 'mtime_ns': stat.st_mtime_ns, 'size_bytes': stat.st_size,
                'coordinate_extents_um_zyx': [[float(metadata[a]['values'][0])*({'mm':1000.}.get(str(metadata[a]['units']).lower(),1.)),
                                              float(metadata[a]['values'][-1])*({'mm':1000.}.get(str(metadata[a]['units']).lower(),1.))] for a in 'zyx']}
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise UserFacingError(f'Reconstruction sidecar calibration could not be validated: {exc}. Supply a corrected sidecar or explicitly enter layout and all three voxel sizes.') from exc


def prepare_input(config: dict) -> dict:
    from .pipeline import APP_ROOT, UserFacingError, read_tiff_spacing, _rational_to_float
    from pipeline_scripts.tiff_calibration import read_z_spacing_um

    if config.get('input_provenance'):
        return config
    if not config.get('image_path'):
        raise UserFacingError('Select a TIFF image first.')
    path = Path(config['image_path']).expanduser().resolve()
    if not path.is_file():
        raise UserFacingError(f'Image file does not exist: {path}')
    layout = config.get('input_layout', 'metadata')
    if layout not in {'metadata', 'ZYX', 'YZX'}:
        raise UserFacingError('Choose metadata, pages-as-depth (ZYX), or resliced YZX layout.')
    raw_voxels = config.get('voxel_size_um_zyx')
    if raw_voxels is not None and (not isinstance(raw_voxels, (list, tuple)) or len(raw_voxels) != 3):
        raise UserFacingError('Supply exactly three voxel sizes in Z, Y, X order.')
    manual = raw_voxels is not None and any(v not in (None, '') for v in raw_voxels)
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise UserFacingError('Choose a TIFF containing a single grayscale volume, not multiple series.')
        series = tif.series[0]
        axes, shape, dtype = series.axes, series.shape, series.dtype
        if len(shape) != 3 or axes not in {'ZYX', 'IYX', 'YZX'}:
            raise UserFacingError(f'TIFF shape {shape}, axes {axes} is not a supported grayscale volume. Time/channel/color axes cannot be treated as depth.')
        metadata = tif.imagej_metadata or {}
        sidecar = None
        if not manual or layout == 'metadata':
            if axes != 'ZYX' or not metadata.get('spacing'):
                sidecar = read_reconstruction_sidecar(path, shape)
        if layout == 'metadata':
            if sidecar is not None:
                layout = sidecar['layout']
            elif axes == 'ZYX':
                layout = 'ZYX'
            elif axes == 'YZX':
                layout = 'YZX'
            elif axes == 'IYX' and metadata.get('slices') == shape[0] and metadata.get('frames', 1) == 1 and metadata.get('channels', 1) == 1:
                layout = 'ZYX'
            else:
                raise UserFacingError(f'This TIFF reports generic or resliced axes {axes}, shape {shape}. In TIFF layout & voxel calibration, explicitly choose whether pages are depth (ZYX) or rows are depth (YZX). No axis or voxel size will be guessed.')
        if not manual and sidecar is None and axes == 'ZYX' and layout == 'ZYX':
            return config
        if manual:
            try:
                voxels = tuple(float(v) for v in raw_voxels)
            except (TypeError, ValueError) as exc:
                raise UserFacingError('Enter all three voxel sizes: Z, Y and X in microns.') from exc
            if len(voxels) != 3 or not all(math.isfinite(v) and v > 0 for v in voxels):
                raise UserFacingError('Z, Y and X voxel sizes must all be finite positive numbers in microns.')
        elif sidecar is not None:
            voxels = sidecar['voxels']
        elif layout == 'ZYX':
            try:
                z = read_z_spacing_um(metadata)
                voxels = (z, 1/_rational_to_float(tif.pages[0].tags['YResolution'].value),
                          1/_rational_to_float(tif.pages[0].tags['XResolution'].value))
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                raise UserFacingError('This TIFF has no complete physical calibration. Enter Z, Y and X voxel sizes in microns under TIFF layout & voxel calibration. Values of 1 in generic resolution tags are not a calibration.') from exc
        else:
            raise UserFacingError('A resliced YZX input requires explicit Z, Y and X voxel sizes in microns to avoid interpreting the stored row spacing as lateral spacing.')
        if not all(math.isfinite(v) and v > 0 for v in voxels):
            raise UserFacingError('TIFF calibration must contain positive finite voxel sizes.')
    stat = path.stat()
    provenance = {'source_path': str(path), 'source_axes': axes, 'source_shape': list(shape),
                  'source_size_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
                  'interpreted_source_layout': layout, 'normalized_axes': 'ZYX',
                  'voxel_size_um_zyx': list(voxels),
                  'calibration_source': 'manual user entry' if manual else ('reconstruction sidecar' if sidecar else 'TIFF ImageJ metadata'),
                  'sidecar': sidecar,
                  'operation': 'lossless axis permutation and calibration; no intensity filtering'}
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:24]
    folder = APP_ROOT/'.runtime'/'normalized_inputs'/digest
    destination = folder/path.name
    with _LOCK:
        if not destination.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            needed = int(np.prod(shape))*np.dtype(dtype).itemsize
            if shutil.disk_usage(folder).free < needed + 64*1024**2:
                raise UserFacingError(f'Normalizing this TIFF needs approximately {needed/1024**3:.2f} GiB of free storage in the app folder.')
            try:
                source = tifffile.memmap(path)
            except ValueError:
                # Standard multipage TIFFs can be compressed or have separated
                # page offsets. Decode one B-scan at a time, not the full volume.
                source = None
            canonical_shape = shape if layout == 'ZYX' else (shape[1], shape[0], shape[2])
            temporary = folder/'normalizing.tmp.tif'
            target = None
            try:
                target = tifffile.memmap(temporary, shape=canonical_shape, dtype=dtype,
                    imagej=True, bigtiff=needed >= 4*1024**3-32*1024**2,
                    metadata={'axes': 'ZYX', 'unit': 'micron', 'spacing': voxels[0]},
                    resolution=(1/voxels[2], 1/voxels[1]))
                if source is not None:
                    canonical = source if layout == 'ZYX' else np.swapaxes(source, 0, 1)
                    for z in range(canonical.shape[0]):
                        target[z] = canonical[z]
                    del canonical, source
                else:
                    with tifffile.TiffFile(path) as tif:
                        pages = tif.series[0].pages
                        if len(pages) != shape[0] or pages[0].shape != shape[1:]:
                            raise UserFacingError('This TIFF needs a decoder for its storage layout. Export it as a grayscale multipage stack or an uncompressed TIFF.')
                        for index, page in enumerate(pages):
                            try:
                                values = page.asarray()
                            except ValueError as exc:
                                raise UserFacingError(f'TIFF page decoding failed: {exc}. Re-export without compression or install the imagecodecs decoder required by this file.') from exc
                            if layout == 'ZYX':
                                target[index] = values
                            else:
                                target[:,index,:] = values
                target.flush()
                del target
                target = None
                os.replace(temporary, destination)
                (folder/'normalization.json').write_text(json.dumps(provenance, indent=2))
            except Exception:
                if target is not None:
                    del target
                if temporary.exists():
                    temporary.unlink()
                raise
        read_tiff_spacing(destination)
    return {**config, 'image_path': str(destination), 'input_provenance': provenance}
