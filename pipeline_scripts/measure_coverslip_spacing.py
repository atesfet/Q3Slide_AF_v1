#!/usr/bin/env python3
"""Measure the separation of two bright OCT interfaces across a tiled volume."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tiff_calibration import read_z_spacing_um
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import find_peaks


def rational_to_float(value) -> float:
    if isinstance(value, tuple):
        return float(value[0]) / float(value[1])
    return float(value)


def subpixel_peak(profile: np.ndarray, index: int) -> float:
    """Three-point parabolic peak location, bounded to half a sample."""
    if index <= 0 or index >= profile.size - 1:
        return float(index)
    left, center, right = map(float, profile[index - 1:index + 2])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-9:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index) + float(np.clip(offset, -0.5, 0.5))


def block_mean_profiles(volume: np.ndarray, block_y: int, block_x: int) -> np.ndarray:
    nz, ny, nx = volume.shape
    nby, nbx = ny // block_y, nx // block_x
    profiles = np.empty((nz, nby, nbx), dtype=np.float32)
    use_y, use_x = nby * block_y, nbx * block_x
    for z in range(nz):
        plane = np.asarray(volume[z, :use_y, :use_x], dtype=np.float32)
        profiles[z] = plane.reshape(nby, block_y, nbx, block_x).mean(axis=(1, 3))
    return profiles


def detect_global_interfaces(profile: np.ndarray, minimum_separation_px: int) -> tuple[int, int, np.ndarray]:
    smooth = gaussian_filter1d(profile.astype(np.float64), 1.0)
    prominence_floor = max(float(np.ptp(smooth)) * 0.05, 1.0)
    peaks, properties = find_peaks(smooth, prominence=prominence_floor, distance=3)
    if peaks.size < 2:
        raise RuntimeError("Could not identify two distinct reflective interfaces")
    prominences = properties["prominences"]
    best = None
    for i in range(peaks.size):
        for j in range(i + 1, peaks.size):
            if peaks[j] - peaks[i] < minimum_separation_px:
                continue
            score = float(prominences[i] + prominences[j])
            if best is None or score > best[0]:
                best = (score, int(peaks[i]), int(peaks[j]))
    if best is None:
        raise RuntimeError("No interface pair met the requested minimum separation")
    return best[1], best[2], smooth


def detect_local_interfaces(
    profiles: np.ndarray,
    first_center: int,
    second_center: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = gaussian_filter1d(profiles, 1.0, axis=0)
    nz, nby, nbx = smooth.shape
    first = np.full((nby, nbx), np.nan, dtype=np.float32)
    second = np.full_like(first, np.nan)
    quality = np.zeros_like(first)
    w1 = (max(1, first_center - search_radius), min(nz - 1, first_center + search_radius + 1))
    w2 = (max(1, second_center - search_radius), min(nz - 1, second_center + search_radius + 1))
    for row in range(nby):
        for col in range(nbx):
            p = smooth[:, row, col]
            z1 = w1[0] + int(np.argmax(p[w1[0]:w1[1]]))
            z2 = w2[0] + int(np.argmax(p[w2[0]:w2[1]]))
            first[row, col] = subpixel_peak(p, z1)
            second[row, col] = subpixel_peak(p, z2)
            baseline = float(np.percentile(p, 25))
            noise = float(np.median(np.abs(p - np.median(p))) * 1.4826 + 1.0)
            quality[row, col] = min((float(p[z1]) - baseline) / noise,
                                    (float(p[z2]) - baseline) / noise)
    return first, second, quality


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def robust_plane(x_mm: np.ndarray, y_mm: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, int]:
    design = np.column_stack((x_mm, y_mm, np.ones(values.size)))
    keep = np.ones(values.size, dtype=bool)
    for _ in range(6):
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values - design @ coefficients
        residual_center = float(np.median(residual[keep]))
        residual_mad = float(np.median(np.abs(residual[keep] - residual_center)) * 1.4826)
        if residual_mad <= 0:
            break
        keep = np.abs(residual - residual_center) <= 3.0 * residual_mad
    return coefficients, float(np.std(residual[keep], ddof=1)), int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size-um", type=float, default=100.0)
    parser.add_argument("--search-radius-um", type=float, default=24.0)
    parser.add_argument("--minimum-separation-um", type=float, default=20.0)
    parser.add_argument("--minimum-quality", type=float, default=2.0)
    parser.add_argument(
        "--analysis-margin-um", type=float, default=1000.0,
        help="Margin excluded from central-field statistics and wedge fit.",
    )
    parser.add_argument(
        "--refractive-index", type=float,
        help="If supplied, also report physical separation = optical-path separation / n.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(args.input) as tf:
        if tf.series[0].axes != "ZYX":
            raise ValueError(f"Expected ZYX TIFF, found {tf.series[0].axes}")
        metadata = tf.imagej_metadata or {}
        z_um = read_z_spacing_um(metadata)
        page = tf.pages[0]
        x_ppu = rational_to_float(page.tags["XResolution"].value)
        y_ppu = rational_to_float(page.tags["YResolution"].value)
        x_um = 1.0 / x_ppu
        y_um = 1.0 / y_ppu

    volume = tifffile.memmap(args.input)
    block_y = max(1, int(round(args.block_size_um / y_um)))
    block_x = max(1, int(round(args.block_size_um / x_um)))
    profiles = block_mean_profiles(volume, block_y, block_x)
    global_profile = np.percentile(profiles, 95, axis=(1, 2))
    minimum_separation_px = max(1, int(round(args.minimum_separation_um / z_um)))
    first_global, second_global, global_smooth = detect_global_interfaces(
        global_profile, minimum_separation_px)
    search_radius = max(2, int(round(args.search_radius_um / z_um)))
    first_px, second_px, quality = detect_local_interfaces(
        profiles, first_global, second_global, search_radius)

    separation_um = (second_px - first_px) * z_um
    first_um = first_px * z_um
    second_um = second_px * z_um
    raw_valid = np.isfinite(separation_um) & (quality >= args.minimum_quality)
    center = float(np.median(separation_um[raw_valid]))
    mad = float(np.median(np.abs(separation_um[raw_valid] - center)) * 1.4826)
    tolerance = max(4.0 * mad, 2.0 * z_um)
    valid = raw_valid & (np.abs(separation_um - center) <= tolerance)

    # Remove isolated block-scale failures while retaining actual broad spatial trends.
    local_median = median_filter(np.where(valid, separation_um, center), size=3, mode="nearest")
    valid &= np.abs(separation_um - local_median) <= max(3.0 * mad, 2.0 * z_um)
    values = separation_um[valid]
    if values.size == 0:
        raise RuntimeError("No reliable local interface pairs remained after quality control")

    center_y_um = (np.arange(valid.shape[0]) + 0.5) * block_y * y_um
    center_x_um = (np.arange(valid.shape[1]) + 0.5) * block_x * x_um
    grid_x_um, grid_y_um = np.meshgrid(center_x_um, center_y_um)
    central = (
        valid &
        (grid_x_um >= args.analysis_margin_um) &
        (grid_x_um <= volume.shape[2] * x_um - args.analysis_margin_um) &
        (grid_y_um >= args.analysis_margin_um) &
        (grid_y_um <= volume.shape[1] * y_um - args.analysis_margin_um)
    )
    if central.sum() < 10:
        central = valid.copy()
    central_values = separation_um[central]
    plane, plane_residual_sd, plane_count = robust_plane(
        grid_x_um[central] / 1000.0,
        grid_y_um[central] / 1000.0,
        central_values,
    )
    wedge_gradient_um_per_mm = float(np.hypot(plane[0], plane[1]))
    wedge_angle_degrees = float(np.degrees(np.arctan(wedge_gradient_um_per_mm / 1000.0)))

    summary = {
        "input": str(args.input.resolve()),
        "shape_zyx": list(map(int, volume.shape)),
        "voxel_size_um_zyx": [z_um, y_um, x_um],
        "measurement_block_um_yx": [block_y * y_um, block_x * x_um],
        "global_first_interface_z_um": float(first_global * z_um),
        "global_second_interface_z_um": float(second_global * z_um),
        "global_peak_separation_um": float((second_global - first_global) * z_um),
        "valid_block_count": int(valid.sum()),
        "total_block_count": int(valid.size),
        "valid_fraction": float(valid.mean()),
        "optical_path_separation_um": distribution_summary(values),
        "central_field": {
            "excluded_margin_um": float(args.analysis_margin_um),
            "valid_block_count": int(central.sum()),
            "optical_path_separation_um": distribution_summary(central_values),
        },
        "wedge_fit_central_field": {
            "model": "separation_um = slope_x * x_mm + slope_y * y_mm + intercept",
            "slope_x_um_per_mm": float(plane[0]),
            "slope_y_um_per_mm": float(plane[1]),
            "intercept_um": float(plane[2]),
            "gradient_um_per_mm": wedge_gradient_um_per_mm,
            "relative_angle_degrees_in_scanner_coordinates": wedge_angle_degrees,
            "residual_standard_deviation_um": plane_residual_sd,
            "inlier_block_count": plane_count,
        },
        "interpretation": (
            "Distances are in the TIFF scanner's axial coordinate. If that coordinate is "
            "air-equivalent optical path length, divide by the intervening medium's "
            "group refractive index to obtain physical distance."
        ),
    }
    if args.refractive_index:
        if args.refractive_index <= 0:
            raise ValueError("--refractive-index must be positive")
        summary["refractive_index"] = float(args.refractive_index)
        summary["physical_separation_um"] = {
            key: value / args.refractive_index
            for key, value in summary["optical_path_separation_um"].items()
        }
        summary["central_field"]["physical_separation_um"] = {
            key: value / args.refractive_index
            for key, value in summary["central_field"]["optical_path_separation_um"].items()
        }

    with (args.output_dir / "coverslip_spacing_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    with (args.output_dir / "coverslip_spacing_blocks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_row", "block_col", "center_y_um", "center_x_um",
                         "first_interface_z_um", "second_interface_z_um",
                         "separation_um", "quality", "valid"])
        for row in range(valid.shape[0]):
            for col in range(valid.shape[1]):
                writer.writerow([
                    row, col, (row + 0.5) * block_y * y_um,
                    (col + 0.5) * block_x * x_um,
                    float(first_um[row, col]), float(second_um[row, col]),
                    float(separation_um[row, col]), float(quality[row, col]),
                    int(valid[row, col]),
                ])

    for name, array in [
        ("first_interface_z_um.tif", first_um),
        ("second_interface_z_um.tif", second_um),
        ("coverslip_separation_um.tif", np.where(valid, separation_um, np.nan)),
        ("detection_quality.tif", quality),
    ]:
        tifffile.imwrite(
            args.output_dir / name, np.asarray(array, dtype=np.float32),
            imagej=True, metadata={"axes": "YX", "unit": "micron"},
            resolution=(1.0 / (block_x * x_um), 1.0 / (block_y * y_um)),
        )

    extent = [0, valid.shape[1] * block_x * x_um,
              valid.shape[0] * block_y * y_um, 0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    depth_axis = np.arange(global_smooth.size) * z_um
    axes[0, 0].plot(depth_axis, global_smooth, color="black")
    axes[0, 0].axvline(first_global * z_um, color="tab:blue", label="interface 1")
    axes[0, 0].axvline(second_global * z_um, color="tab:orange", label="interface 2")
    axes[0, 0].set(xlabel="OCT depth (µm)", ylabel="95th-percentile signal",
                   title="Global depth profile")
    axes[0, 0].legend()
    im = axes[0, 1].imshow(first_um, extent=extent, cmap="viridis")
    axes[0, 1].set_title("First interface depth (µm)")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.85)
    im = axes[0, 2].imshow(second_um, extent=extent, cmap="viridis")
    axes[0, 2].set_title("Second interface depth (µm)")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.85)
    lo, hi = np.percentile(values, [2, 98])
    im = axes[1, 0].imshow(np.where(valid, separation_um, np.nan), extent=extent,
                           cmap="magma", vmin=lo, vmax=hi)
    axes[1, 0].set_title("Interface separation (µm)")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.85)
    axes[1, 1].hist(values, bins=40, color="0.25")
    axes[1, 1].axvline(np.median(values), color="tab:red",
                       label=f"median {np.median(values):.2f} µm")
    axes[1, 1].set(xlabel="Separation (µm)", ylabel="Blocks",
                   title="Valid local measurements")
    axes[1, 1].legend()
    xz = np.mean(profiles, axis=1)
    vlo, vhi = np.percentile(xz, [2, 99.5])
    axes[1, 2].imshow(xz, aspect="auto", cmap="gray", vmin=vlo, vmax=vhi,
                      extent=[0, valid.shape[1] * block_x * x_um,
                              (xz.shape[0] - 1) * z_um, 0])
    axes[1, 2].plot((np.arange(valid.shape[1]) + 0.5) * block_x * x_um,
                    np.nanmedian(np.where(valid, first_um, np.nan), axis=0),
                    color="cyan", lw=1)
    axes[1, 2].plot((np.arange(valid.shape[1]) + 0.5) * block_x * x_um,
                    np.nanmedian(np.where(valid, second_um, np.nan), axis=0),
                    color="orange", lw=1)
    axes[1, 2].set(xlabel="X (µm)", ylabel="OCT depth (µm)",
                   title="Mean X–Z section with detections")
    for ax in [axes[0, 1], axes[0, 2], axes[1, 0]]:
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
    fig.suptitle(f"Coverslip-interface spacing: {args.input.name}")
    fig.savefig(args.output_dir / "coverslip_spacing_qc.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
