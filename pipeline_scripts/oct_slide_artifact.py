#!/usr/bin/env python3
"""Utilities for characterizing and suppressing slide artifacts in OCT mosaics.

The TIFF is treated as a Z,Y,X volume. Mosaic patch coordinates are always
``(row, column)`` and are zero based.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
if __package__:
    from .tiff_calibration import read_z_spacing_um
else:
    from tiff_calibration import read_z_spacing_um
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates


@dataclass(frozen=True)
class VolumeGeometry:
    shape_zyx: tuple[int, int, int]
    dtype: str
    rows: int
    columns: int
    patch_shape_yx: tuple[int, int]
    voxel_size_um_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class ArtifactModel:
    profile: np.ndarray
    radial_gain: np.ndarray
    radial_z_offset: np.ndarray
    center_z: float
    support_radius_z: int
    baseline_quantile: float
    reference_patches: tuple[tuple[int, int], ...]
    reference_centers_z: tuple[float, ...]
    reference_peak_excess: tuple[float, ...]
    voxel_size_um_zyx: tuple[float, float, float]
    reference_patch_shape_yx: tuple[int, int]


def parse_patch(value: str) -> tuple[int, int]:
    try:
        row, column = (int(v.strip()) for v in value.split(","))
    except Exception as exc:
        raise ValueError(f"Patch must be ROW,COLUMN; got {value!r}") from exc
    return row, column


def open_volume(path: Path) -> np.ndarray:
    array = tifffile.memmap(path)
    if array.ndim != 3:
        raise ValueError(f"Expected one Z,Y,X volume, got shape {array.shape}")
    return array


def read_geometry(
    path: Path,
    rows: int | None = None,
    columns: int | None = None,
    patch_size_um: float = 1000.0,
) -> VolumeGeometry:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.shape) != 3:
            raise ValueError(f"Expected Z,Y,X TIFF; got {series.shape} ({series.axes})")
        z_count, y_count, x_count = (int(v) for v in series.shape)
        metadata = tif.imagej_metadata or {}
        z_um = read_z_spacing_um(metadata)
        page = tif.pages[0]
        x_res = page.tags.get("XResolution")
        y_res = page.tags.get("YResolution")

        def spacing_from_resolution(tag, fallback: float | None) -> float | None:
            if tag is None:
                return fallback
            numerator, denominator = tag.value
            pixels_per_unit = float(numerator) / float(denominator)
            return 1.0 / pixels_per_unit if pixels_per_unit > 0 else fallback

        # The ImageJ unit is micron and TIFF resolution is pixels per that unit.
        x_um = spacing_from_resolution(
            x_res, patch_size_um * columns / x_count if columns is not None else None
        )
        y_um = spacing_from_resolution(
            y_res, patch_size_um * rows / y_count if rows is not None else None
        )

        def infer_count(
            supplied: int | None, pixel_count: int, spacing_um: float | None, axis: str
        ) -> int:
            if supplied is not None:
                if supplied <= 0:
                    raise ValueError(f"{axis} patch count must be positive")
                return supplied
            if spacing_um is None:
                raise ValueError(
                    f"Cannot infer {axis} patch count because lateral pixel spacing is "
                    f"missing; provide --{'rows' if axis == 'Y' else 'columns'}"
                )
            estimate = pixel_count * spacing_um / patch_size_um
            inferred = int(round(estimate))
            if inferred <= 0 or abs(estimate - inferred) > 0.05:
                raise ValueError(
                    f"Cannot reliably infer {axis} patch count: physical extent implies "
                    f"{estimate:.4f} patches of {patch_size_um:g} um; provide a manual override"
                )
            return inferred

        rows = infer_count(rows, y_count, y_um, "Y")
        columns = infer_count(columns, x_count, x_um, "X")
        if y_count % rows or x_count % columns:
            raise ValueError(
                f"Volume {series.shape} is not divisible by grid {rows}x{columns}"
            )
        # Manual grid values provide a safe fallback when TIFF resolution tags are absent.
        if x_um is None:
            x_um = patch_size_um * columns / x_count
        if y_um is None:
            y_um = patch_size_um * rows / y_count
        return VolumeGeometry(
            shape_zyx=(z_count, y_count, x_count),
            dtype=str(series.dtype),
            rows=rows,
            columns=columns,
            patch_shape_yx=(y_count // rows, x_count // columns),
            voxel_size_um_zyx=(z_um, y_um, x_um),
        )


def read_lateral_grid(
    path: Path,
    rows: int | None = None,
    columns: int | None = None,
    patch_size_um: float = 1000.0,
) -> tuple[int, int, float, float]:
    """Return rows, columns, Y spacing, and X spacing for a 2-D/3-D TIFF."""
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
        if len(shape) < 2:
            raise ValueError(f"Expected at least a Y,X TIFF; got {shape}")
        y_count, x_count = (int(v) for v in shape[-2:])
        page = tif.pages[0]

        def spacing(tag) -> float | None:
            if tag is None:
                return None
            numerator, denominator = tag.value
            pixels_per_unit = float(numerator) / float(denominator)
            return 1.0 / pixels_per_unit if pixels_per_unit > 0 else None

        x_um = spacing(page.tags.get("XResolution"))
        y_um = spacing(page.tags.get("YResolution"))

    def resolve(
        supplied: int | None, pixels: int, pixel_um: float | None, option: str
    ) -> int:
        if supplied is not None:
            if supplied <= 0:
                raise ValueError(f"--{option} must be positive")
            return supplied
        if pixel_um is None:
            raise ValueError(
                f"Cannot infer --{option}: TIFF lateral pixel spacing is missing"
            )
        estimate = pixels * pixel_um / patch_size_um
        inferred = int(round(estimate))
        if inferred <= 0 or abs(estimate - inferred) > 0.05:
            raise ValueError(
                f"Cannot reliably infer --{option}: physical extent implies "
                f"{estimate:.4f} patches of {patch_size_um:g} um"
            )
        return inferred

    rows = resolve(rows, y_count, y_um, "rows")
    columns = resolve(columns, x_count, x_um, "columns")
    if y_count % rows or x_count % columns:
        raise ValueError(f"Image {shape} is not divisible by grid {rows}x{columns}")
    if y_um is None:
        y_um = patch_size_um * rows / y_count
    if x_um is None:
        x_um = patch_size_um * columns / x_count
    return rows, columns, y_um, x_um


def validate_patch(patch: tuple[int, int], geometry: VolumeGeometry) -> None:
    row, column = patch
    if not (0 <= row < geometry.rows and 0 <= column < geometry.columns):
        raise ValueError(
            f"Patch {patch} is outside zero-based grid "
            f"rows 0..{geometry.rows - 1}, columns 0..{geometry.columns - 1}"
        )


def extract_patch(
    volume: np.ndarray, geometry: VolumeGeometry, patch: tuple[int, int]
) -> np.ndarray:
    validate_patch(patch, geometry)
    row, column = patch
    patch_y, patch_x = geometry.patch_shape_yx
    return np.asarray(
        volume[:, row * patch_y : (row + 1) * patch_y,
               column * patch_x : (column + 1) * patch_x]
    )


def robust_depth_profile(patch_zyx: np.ndarray) -> np.ndarray:
    """Median depth profile, robust to sparse dust and tissue."""
    return np.median(np.asarray(patch_zyx, dtype=np.float32), axis=(1, 2))


def depth_baseline(profile: np.ndarray, quantile: float = 0.2) -> float:
    return float(np.quantile(profile, quantile))


def parabolic_peak(profile: np.ndarray) -> float:
    index = int(np.argmax(profile))
    if index == 0 or index == len(profile) - 1:
        return float(index)
    left, center, right = (float(profile[index + d]) for d in (-1, 0, 1))
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)


def normalized_profile(
    patch_zyx: np.ndarray, baseline_quantile: float = 0.2
) -> tuple[np.ndarray, float, float]:
    profile = gaussian_filter1d(robust_depth_profile(patch_zyx), sigma=0.7)
    baseline = depth_baseline(profile, baseline_quantile)
    excess = np.maximum(profile - baseline, 0.0)
    peak = float(excess.max())
    if peak <= 0:
        raise ValueError("Reference patch has no positive axial artifact profile")
    return excess / peak, parabolic_peak(excess), peak


def shift_profile(profile: np.ndarray, shift: float, output_z: np.ndarray) -> np.ndarray:
    source_z = output_z - shift
    return np.interp(source_z, np.arange(len(profile)), profile, left=0.0, right=0.0)


def build_artifact_model(
    reference_volumes: Sequence[np.ndarray],
    reference_patches: Sequence[tuple[int, int]],
    voxel_size_um_zyx: tuple[float, float, float],
    baseline_quantile: float = 0.2,
    support_fraction: float = 0.03,
) -> ArtifactModel:
    profiles: list[np.ndarray] = []
    centers: list[float] = []
    peaks: list[float] = []
    for patch_volume in reference_volumes:
        profile, center, peak = normalized_profile(patch_volume, baseline_quantile)
        profiles.append(profile)
        centers.append(center)
        peaks.append(peak)
    common_center = float(np.median(centers))
    z_axis = np.arange(len(profiles[0]), dtype=np.float32)
    aligned = [shift_profile(p, common_center - c, z_axis) for p, c in zip(profiles, centers)]
    model_profile = np.median(np.stack(aligned), axis=0)
    model_profile /= max(float(model_profile.max()), np.finfo(np.float32).eps)
    support = np.flatnonzero(model_profile >= support_fraction)
    radius = int(np.ceil(max(abs(support - common_center)))) if support.size else 8
    patch_shape = tuple(int(v) for v in reference_volumes[0].shape[1:])
    maximum_radius = int(np.ceil(np.hypot(patch_shape[0] / 2, patch_shape[1] / 2)))
    preliminary = ArtifactModel(
        profile=model_profile.astype(np.float32),
        radial_gain=np.ones(maximum_radius + 1, dtype=np.float32),
        radial_z_offset=np.zeros(maximum_radius + 1, dtype=np.float32),
        center_z=common_center,
        support_radius_z=radius,
        baseline_quantile=baseline_quantile,
        reference_patches=tuple(reference_patches),
        reference_centers_z=tuple(float(v) for v in centers),
        reference_peak_excess=tuple(float(v) for v in peaks),
        voxel_size_um_zyx=voxel_size_um_zyx,
        reference_patch_shape_yx=patch_shape,
    )
    radial_curves = [_estimate_radial_gain(v, preliminary) for v in reference_volumes]
    radial_gain = np.median(np.stack(radial_curves), axis=0).astype(np.float32)
    radial_gain /= max(float(np.median(radial_gain)), np.finfo(np.float32).eps)
    radial_gain = np.clip(radial_gain, 0.65, 1.35)
    radial_z_curves = [_estimate_radial_z_offset(v, preliminary) for v in reference_volumes]
    radial_z_offset = np.median(np.stack(radial_z_curves), axis=0).astype(np.float32)
    radial_z_offset -= float(np.median(radial_z_offset))
    radial_z_offset = np.clip(radial_z_offset, -1.5, 1.5)
    return ArtifactModel(
        **{
            **preliminary.__dict__,
            "radial_gain": radial_gain,
            "radial_z_offset": radial_z_offset,
        }
    )


def save_model(model: ArtifactModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "artifact_profile.npy", model.profile)
    np.save(output_dir / "radial_gain.npy", model.radial_gain)
    np.save(output_dir / "radial_z_offset.npy", model.radial_z_offset)
    payload = asdict(model)
    payload.pop("profile")
    payload.pop("radial_gain")
    payload.pop("radial_z_offset")
    payload["reference_patches"] = [list(v) for v in model.reference_patches]
    payload["profile_file"] = "artifact_profile.npy"
    payload["radial_gain_file"] = "radial_gain.npy"
    payload["radial_z_offset_file"] = "radial_z_offset.npy"
    with (output_dir / "artifact_model.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_model(model_dir: Path) -> ArtifactModel:
    with (model_dir / "artifact_model.json").open() as handle:
        payload = json.load(handle)
    profile = np.load(model_dir / payload.pop("profile_file"))
    radial_gain = np.load(model_dir / payload.pop("radial_gain_file"))
    radial_z_offset = np.load(model_dir / payload.pop("radial_z_offset_file"))
    return ArtifactModel(
        profile=np.asarray(profile, dtype=np.float32),
        radial_gain=np.asarray(radial_gain, dtype=np.float32),
        radial_z_offset=np.asarray(radial_z_offset, dtype=np.float32),
        center_z=float(payload["center_z"]),
        support_radius_z=int(payload["support_radius_z"]),
        baseline_quantile=float(payload["baseline_quantile"]),
        reference_patches=tuple(tuple(v) for v in payload["reference_patches"]),
        reference_centers_z=tuple(payload["reference_centers_z"]),
        reference_peak_excess=tuple(payload["reference_peak_excess"]),
        voxel_size_um_zyx=tuple(payload["voxel_size_um_zyx"]),
        reference_patch_shape_yx=tuple(payload["reference_patch_shape_yx"]),
    )


def estimate_patch_center(patch_zyx: np.ndarray, model: ArtifactModel) -> float:
    """Estimate the slide center using normalized profile correlation."""
    profile = gaussian_filter1d(robust_depth_profile(patch_zyx), sigma=0.7)
    profile = np.maximum(profile - depth_baseline(profile, model.baseline_quantile), 0.0)
    template = model.profile - model.profile.mean()
    signal = profile - profile.mean()
    correlation = np.correlate(signal, template, mode="full")
    # Parabolic refinement avoids quantizing patch alignment to the 1.434 µm
    # axial sampling interval.
    lag = parabolic_peak(correlation) - (len(template) - 1)
    return float(np.clip(model.center_z + lag, 0, len(profile) - 1))


def _polynomial_design(y: np.ndarray, x: np.ndarray, degree: int) -> np.ndarray:
    terms = []
    for total_degree in range(degree + 1):
        for y_degree in range(total_degree + 1):
            x_degree = total_degree - y_degree
            terms.append((y ** y_degree) * (x ** x_degree))
    return np.column_stack(terms)


def robust_polynomial_surface(
    values: np.ndarray,
    degree: int,
    sample_step: int = 6,
    iterations: int = 6,
    clip_low: float = 3.0,
    clip_high: float = 2.5,
) -> np.ndarray:
    """Fit a smooth 2-D field while rejecting tissue/dust outliers."""
    height, width = values.shape
    y_full, x_full = np.indices(values.shape, dtype=np.float32)
    y_normalized = 2.0 * y_full / max(height - 1, 1) - 1.0
    x_normalized = 2.0 * x_full / max(width - 1, 1) - 1.0
    y_sample = y_normalized[::sample_step, ::sample_step].ravel()
    x_sample = x_normalized[::sample_step, ::sample_step].ravel()
    target = values[::sample_step, ::sample_step].astype(np.float64).ravel()
    design = _polynomial_design(y_sample, x_sample, degree)
    keep = np.isfinite(target)
    coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    for _ in range(iterations):
        residual = target - design @ coefficients
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        sigma = max(1.4826 * mad, 1e-3)
        new_keep = keep & (residual >= center - clip_low * sigma) & (
            residual <= center + clip_high * sigma
        )
        if new_keep.sum() < design.shape[1] * 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    full_design = _polynomial_design(y_normalized.ravel(), x_normalized.ravel(), degree)
    return (full_design @ coefficients).reshape(values.shape).astype(np.float32)


def _artifact_shape(
    model: ArtifactModel, surface_z: np.ndarray, z_count: int
) -> np.ndarray:
    z_grid = np.arange(z_count, dtype=np.float32)[:, None, None]
    template_coordinate = z_grid - surface_z[None, :, :] + model.center_z
    return map_coordinates(
        model.profile,
        [template_coordinate],
        order=1,
        mode="constant",
        cval=0.0,
    )


def _amplitude_from_shape(
    data: np.ndarray, artifact_shape: np.ndarray, baseline_quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.quantile(data, baseline_quantile, axis=0).astype(np.float32)
    excess = np.maximum(data - baseline[None, :, :], 0.0)
    numerator = np.sum(excess * artifact_shape, axis=0)
    denominator = np.sum(artifact_shape * artifact_shape, axis=0) + 1e-6
    return (numerator / denominator).astype(np.float32), baseline


def _radial_coordinates(shape_yx: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape_yx, dtype=np.float32)
    center_y = (shape_yx[0] - 1) / 2.0
    center_x = (shape_yx[1] - 1) / 2.0
    return np.hypot(y - center_y, x - center_x)


def _estimate_radial_gain(patch_zyx: np.ndarray, model: ArtifactModel) -> np.ndarray:
    data = np.asarray(patch_zyx, dtype=np.float32)
    center = estimate_patch_center(data, model)
    surface = estimate_surface_map(data, model, center)
    shape = _artifact_shape(model, surface, data.shape[0])
    raw_amplitude, _ = _amplitude_from_shape(data, shape, model.baseline_quantile)
    smooth_amplitude = robust_polynomial_surface(
        gaussian_filter(raw_amplitude, sigma=3.0), degree=4, sample_step=8,
        clip_low=3.0, clip_high=1.75
    )
    ratio = raw_amplitude / np.maximum(smooth_amplitude, 1.0)
    radius = _radial_coordinates(raw_amplitude.shape)
    bins = np.floor(radius).astype(np.int32)
    curve = np.ones(len(model.radial_gain), dtype=np.float32)
    for index in range(len(curve)):
        values = ratio[bins == index]
        if values.size:
            curve[index] = np.median(values)
        elif index:
            curve[index] = curve[index - 1]
    curve = gaussian_filter1d(curve, sigma=0.65)
    curve /= max(float(np.median(curve)), np.finfo(np.float32).eps)
    return np.clip(curve, 0.65, 1.35)


def _estimate_radial_z_offset(patch_zyx: np.ndarray, model: ArtifactModel) -> np.ndarray:
    """Estimate repeatable scanner-centered axial ripple (Newton-ring pattern)."""
    data = np.asarray(patch_zyx, dtype=np.float32)
    center = estimate_patch_center(data, model)
    smoothed = gaussian_filter(data, sigma=(0.65, 2.0, 2.0))
    baseline = np.quantile(smoothed, model.baseline_quantile, axis=0)
    signal = np.maximum(smoothed - baseline[None, :, :], 0.0)
    low = max(0, int(np.floor(center - 5)))
    high = min(signal.shape[0], int(np.ceil(center + 6)))
    peak = np.argmax(signal[low:high], axis=0) + low
    y, x = np.indices(peak.shape)
    peak_value = signal[peak, y, x]
    left = signal[np.maximum(peak - 1, 0), y, x]
    right = signal[np.minimum(peak + 1, signal.shape[0] - 1), y, x]
    denominator = left - 2.0 * peak_value + right
    offset = np.zeros_like(peak_value, dtype=np.float32)
    valid = np.abs(denominator) > 1e-6
    offset[valid] = 0.5 * (left[valid] - right[valid]) / denominator[valid]
    raw_surface = peak.astype(np.float32) + np.clip(offset, -0.5, 0.5)
    smooth_surface = robust_polynomial_surface(
        raw_surface, degree=2, sample_step=6, clip_low=2.5, clip_high=2.5
    )
    residual = raw_surface - smooth_surface
    radius = _radial_coordinates(raw_surface.shape)
    bins = np.floor(radius).astype(np.int32)
    curve = np.zeros(len(model.radial_z_offset), dtype=np.float32)
    for index in range(len(curve)):
        values = residual[bins == index]
        if values.size:
            curve[index] = np.median(values)
        elif index:
            curve[index] = curve[index - 1]
    curve = gaussian_filter1d(curve, sigma=0.65)
    curve -= float(np.median(curve))
    return np.clip(curve, -1.5, 1.5)


def estimate_surface_map(
    patch_zyx: np.ndarray,
    model: ArtifactModel,
    patch_center: float | None = None,
    xy_sigma_px: float = 12.0,
    search_radius_z: int = 5,
) -> np.ndarray:
    """Estimate a smooth, sub-patch slide surface from locally averaged A-scans."""
    data = np.asarray(patch_zyx, dtype=np.float32)
    if patch_center is None:
        patch_center = estimate_patch_center(data, model)
    smoothed = gaussian_filter(data, sigma=(0.7, xy_sigma_px, xy_sigma_px))
    baseline = np.quantile(smoothed, model.baseline_quantile, axis=0)
    signal = np.maximum(smoothed - baseline[None, :, :], 0.0)
    low = max(0, int(np.floor(patch_center - search_radius_z)))
    high = min(signal.shape[0], int(np.ceil(patch_center + search_radius_z + 1)))
    local = signal[low:high]
    local_argmax = np.argmax(local, axis=0) + low

    # Sub-voxel parabolic refinement at each lateral pixel.
    y_grid, x_grid = np.indices(local_argmax.shape)
    center_value = signal[local_argmax, y_grid, x_grid]
    left_index = np.maximum(local_argmax - 1, 0)
    right_index = np.minimum(local_argmax + 1, signal.shape[0] - 1)
    left_value = signal[left_index, y_grid, x_grid]
    right_value = signal[right_index, y_grid, x_grid]
    denominator = left_value - 2.0 * center_value + right_value
    offset = np.zeros_like(center_value, dtype=np.float32)
    valid = np.abs(denominator) > 1e-6
    offset[valid] = 0.5 * (left_value[valid] - right_value[valid]) / denominator[valid]
    offset = np.clip(offset, -0.5, 0.5)
    raw_surface = gaussian_filter(
        local_argmax.astype(np.float32) + offset, sigma=max(xy_sigma_px / 4, 1.0)
    )
    # A slide is locally well represented by a gently curved quadratic. Robust
    # fitting prevents a broad tissue region from becoming a false slide surface.
    surface = robust_polynomial_surface(
        raw_surface, degree=2, sample_step=max(3, int(xy_sigma_px / 2)),
        clip_low=2.5, clip_high=2.5
    )
    surface += float(patch_center) - float(np.median(surface))
    radius = _radial_coordinates(surface.shape)
    radial_z_offset = np.interp(
        radius,
        np.arange(len(model.radial_z_offset), dtype=np.float32),
        model.radial_z_offset,
    ).astype(np.float32)
    surface += radial_z_offset
    return np.clip(
        surface,
        patch_center - search_radius_z,
        patch_center + search_radius_z,
    ).astype(np.float32)


def synthesize_artifact(
    patch_zyx: np.ndarray,
    model: ArtifactModel,
    surface_z: np.ndarray,
    lateral_sigma_px: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a shifted/scaled smooth artifact and return artifact, baseline, amplitude."""
    data = np.asarray(patch_zyx, dtype=np.float32)
    baseline = np.quantile(data, model.baseline_quantile, axis=0).astype(np.float32)
    artifact_shape = _artifact_shape(model, surface_z, data.shape[0])
    fitted_amplitude, baseline = _amplitude_from_shape(
        data, artifact_shape, model.baseline_quantile
    )
    raw_amplitude = gaussian_filter(fitted_amplitude, sigma=max(lateral_sigma_px / 4, 1.0))
    # The glass illumination/vignetting is smooth. A robust fourth-order field
    # follows that shape but rejects high positive tissue and dust contributions.
    amplitude = robust_polynomial_surface(
        raw_amplitude,
        degree=4,
        sample_step=max(4, int(lateral_sigma_px / 2)),
        clip_low=3.0,
        clip_high=1.75,
    )
    amplitude = np.maximum(amplitude, 0.0)
    radius = _radial_coordinates(amplitude.shape)
    radial_gain = np.interp(
        radius,
        np.arange(len(model.radial_gain), dtype=np.float32),
        model.radial_gain,
    ).astype(np.float32)
    amplitude *= radial_gain
    return artifact_shape * amplitude[None, :, :], baseline, amplitude


def correct_patch(
    patch_zyx: np.ndarray,
    model: ArtifactModel,
    strength: float = 0.8,
    xy_sigma_px: float = 12.0,
    amplitude_sigma_px: float = 18.0,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Conservatively subtract the registered, smooth glass component.

    Values are floored at the locally estimated acquisition baseline. This avoids
    introducing dark bands. Random speckle and local scratches are intentionally
    not copied from the reference patch.
    """
    data = np.asarray(patch_zyx, dtype=np.float32)
    center = estimate_patch_center(data, model)
    surface = estimate_surface_map(data, model, center, xy_sigma_px=xy_sigma_px)
    artifact, baseline, amplitude = synthesize_artifact(
        data, model, surface, lateral_sigma_px=amplitude_sigma_px
    )
    corrected = np.maximum(data - float(strength) * artifact, baseline[None, :, :])
    corrected = np.clip(np.rint(corrected), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    diagnostics: dict[str, np.ndarray | float] = {
        "center_z": center,
        "surface_z": surface,
        "baseline": baseline,
        "amplitude": amplitude,
        "artifact": artifact,
    }
    return corrected, diagnostics


def patch_qc_rows(
    volume: np.ndarray, geometry: VolumeGeometry
) -> Iterable[dict[str, float | int]]:
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            patch = extract_patch(volume, geometry, (row, column)).astype(np.float32)
            profile = robust_depth_profile(patch)
            baseline = depth_baseline(profile)
            yield {
                "row": row,
                "column": column,
                "median_profile_peak_z": parabolic_peak(profile),
                "median_profile_peak_excess": float(profile.max() - baseline),
                "median_baseline": baseline,
                "deep_tail_p99_excess": float(
                    np.quantile(patch[max(0, patch.shape[0] - 8) :], 0.99) - baseline
                ),
                "minimum": int(patch.min()),
                "maximum": int(patch.max()),
                "mean": float(patch.mean()),
                "standard_deviation": float(patch.std()),
            }


def estimate_background_patches(
    volume: np.ndarray,
    geometry: VolumeGeometry,
    count: int = 10,
    tissue_mask: np.ndarray | None = None,
    maximum_tissue_fraction: float = 0.02,
) -> tuple[list[tuple[int, int]], list[dict[str, float | int | bool | str]]]:
    """Rank likely blank patches, optionally using a previously generated mask.

    With a mask, selection is based primarily on tissue occupancy and is suitable
    for refinement. Without one, low deep-tail signal and low patch variance form
    a conservative bootstrap; the report labels that evidence as lower confidence.
    """
    qc = list(patch_qc_rows(volume, geometry))
    tail = np.array([float(v["deep_tail_p99_excess"]) for v in qc], dtype=np.float64)
    spread = np.array([float(v["standard_deviation"]) for v in qc], dtype=np.float64)

    def robust_z(values: np.ndarray) -> np.ndarray:
        center = np.median(values)
        scale = 1.4826 * np.median(np.abs(values - center)) + 1e-6
        return (values - center) / scale

    raw_score = robust_z(np.log1p(np.maximum(tail, 0))) + robust_z(spread)
    occupancies = np.full(len(qc), np.nan, dtype=np.float64)
    if tissue_mask is not None:
        if tissue_mask.shape != geometry.shape_zyx[1:]:
            raise ValueError(
                f"Tissue mask shape {tissue_mask.shape} does not match {geometry.shape_zyx[1:]}"
            )
        patch_y, patch_x = geometry.patch_shape_yx
        for index, item in enumerate(qc):
            row, column = int(item["row"]), int(item["column"])
            occupancies[index] = np.mean(
                tissue_mask[
                    row * patch_y : (row + 1) * patch_y,
                    column * patch_x : (column + 1) * patch_x,
                ] > 0
            )
        eligible = np.flatnonzero(occupancies <= maximum_tissue_fraction)
        if eligible.size < min(3, count):
            eligible = np.argsort(occupancies)[: max(3, count)]
        order = eligible[np.lexsort((raw_score[eligible], occupancies[eligible]))]
        basis = "tissue-mask occupancy plus raw-volume tie-break"
    else:
        order = np.argsort(raw_score)
        basis = "raw-volume bootstrap (deep-tail signal plus patch variance)"
    selected_indices = set(int(v) for v in order[: min(count, len(order))])
    selected = [
        (int(qc[index]["row"]), int(qc[index]["column"]))
        for index in order[: min(count, len(order))]
    ]
    report: list[dict[str, float | int | bool | str]] = []
    for index, item in enumerate(qc):
        occupancy = occupancies[index]
        if np.isfinite(occupancy):
            confidence = (
                "high" if occupancy <= maximum_tissue_fraction / 2
                else "medium" if occupancy <= maximum_tissue_fraction
                else "low"
            )
        else:
            confidence = "bootstrap"
        report.append(
            {
                "row": int(item["row"]),
                "column": int(item["column"]),
                "selected_as_background": index in selected_indices,
                "confidence": confidence,
                "selection_basis": basis,
                "tissue_fraction": float(occupancy) if np.isfinite(occupancy) else "",
                "raw_background_score": float(raw_score[index]),
                "deep_tail_p99_excess": float(tail[index]),
                "standard_deviation": float(spread[index]),
            }
        )
    report.sort(key=lambda v: (not bool(v["selected_as_background"]), float(v["raw_background_score"])))
    return selected, report


def write_csv(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
