#!/usr/bin/env python3
"""Generate a tissue-preserving 2-D OCT projection directly from the raw mosaic.

The fitted glass amplitude in a tissue patch is capped by an envelope learned
only from blank patches. Tissue detection uses blank-calibrated, depth-contiguous
evidence, while the primary projection retains signal above the blank median.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oct_mpl_cache")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    convolve1d,
    gaussian_filter,
    label,
    map_coordinates,
)

from oct_slide_artifact import (
    correct_patch,
    extract_patch,
    load_model,
    open_volume,
    parse_patch,
    read_geometry,
    validate_patch,
)


def write_imagej(path: Path, array: np.ndarray, y_um: float, x_um: float) -> None:
    tifffile.imwrite(
        path,
        array,
        imagej=True,
        metadata={"axes": "YX", "unit": "micron"},
        resolution=(1.0 / x_um, 1.0 / y_um),
    )


def flatten_excess(
    data: np.ndarray,
    baseline: np.ndarray,
    shift: np.ndarray,
    output_z: int,
) -> np.ndarray:
    """Flatten baseline-subtracted signal using a precomputed surface shift."""
    excess = np.maximum(data.astype(np.float32) - baseline[None], 0.0)
    output_axis = np.arange(output_z, dtype=np.float32)[:, None, None]
    source_z = output_axis - shift[None]
    yy, xx = np.indices(shift.shape, dtype=np.float32)
    return map_coordinates(
        excess,
        [source_z, np.broadcast_to(yy, source_z.shape), np.broadcast_to(xx, source_z.shape)],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def safe_correct(
    patch: np.ndarray,
    diagnostics: dict,
    blank_median_amplitude: np.ndarray,
    blank_upper_amplitude: np.ndarray,
    scale_quantile: float,
    minimum_scale: float,
    maximum_scale: float,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Subtract glass while preventing tissue from inflating its fitted amplitude."""
    fitted = np.asarray(diagnostics["amplitude"], dtype=np.float32)
    ratio = fitted / np.maximum(blank_median_amplitude, 1.0)
    anchor_scale = float(
        np.clip(np.percentile(ratio, scale_quantile), minimum_scale, maximum_scale)
    )
    cap = blank_upper_amplitude * anchor_scale
    safe_amplitude = np.minimum(fitted, cap)
    scale = safe_amplitude / np.maximum(fitted, 1e-6)
    artifact = (
        np.asarray(diagnostics["artifact"], dtype=np.float32)
        * scale[None]
        * float(strength)
    )
    baseline = np.asarray(diagnostics["baseline"], dtype=np.float32)
    corrected = np.maximum(patch.astype(np.float32) - artifact, baseline[None])
    cap_engagement = np.clip(1.0 - scale, 0.0, 1.0)
    return corrected, baseline, cap_engagement, anchor_scale


def largest_tissue_mask(
    projection: np.ndarray,
    texture_source: np.ndarray,
    blank_coordinates: list[tuple[int, int]],
    patch_shape: tuple[int, int],
    downsample: int,
    quantile: float,
    neck_opening_iterations: int,
    texture_quantile: float,
    texture_opening_iterations: int,
) -> tuple[np.ndarray, float]:
    low = gaussian_filter(projection[::downsample, ::downsample], sigma=max(1.0, 20 / downsample))
    py, px = patch_shape
    low_py, low_px = py // downsample, px // downsample
    blank_values = np.concatenate(
        [
            low[r * low_py : (r + 1) * low_py, c * low_px : (c + 1) * low_px].ravel()
            for r, c in blank_coordinates
        ]
    )
    threshold = float(np.percentile(blank_values, quantile))
    candidate = low > threshold
    if neck_opening_iterations > 0:
        candidate = binary_opening(candidate, iterations=neck_opening_iterations)
    labels, count = label(candidate)
    sizes = np.bincount(labels.ravel())
    if count == 0 or sizes[1:].max(initial=0) == 0:
        raise RuntimeError("No tissue component detected")
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    mask = binary_closing(mask, iterations=4)
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask)
    if texture_quantile > 0:
        texture_low = texture_source[::downsample, ::downsample]
        highpass = texture_low - gaussian_filter(
            texture_low, sigma=max(1.0, 8.0 / downsample)
        )
        texture = np.sqrt(
            gaussian_filter(highpass * highpass, sigma=max(1.0, 12.0 / downsample))
        )
        blank_texture = np.concatenate(
            [
                texture[
                    r * low_py : (r + 1) * low_py,
                    c * low_px : (c + 1) * low_px,
                ].ravel()
                for r, c in blank_coordinates
            ]
        )
        texture_candidate = texture > np.percentile(blank_texture, texture_quantile)
        if texture_opening_iterations > 0:
            texture_candidate = binary_opening(
                texture_candidate, iterations=texture_opening_iterations
            )
        texture_labels, texture_count = label(texture_candidate)
        texture_sizes = np.bincount(texture_labels.ravel())
        if texture_count and texture_sizes[1:].max(initial=0) > 0:
            texture_sizes[0] = 0
            texture_mask = texture_labels == int(np.argmax(texture_sizes))
            texture_mask = binary_dilation(
                texture_mask, iterations=texture_opening_iterations + 2
            )
            texture_mask = binary_fill_holes(texture_mask)
            combined = mask & texture_mask
            # Reject a texture refinement that accidentally locks onto an
            # isolated high-contrast glass/bubble component.
            if combined.mean() >= 0.5 * mask.mean():
                mask = combined
    mask = np.repeat(np.repeat(mask, downsample, axis=0), downsample, axis=1)
    return mask[: projection.shape[0], : projection.shape[1]], threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path, help="Original uncorrected ZYX OCT TIFF")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--surface-maps", type=Path, required=True)
    parser.add_argument(
        "--patch-interface-bounds",
        type=Path,
        help="Optional per-patch upper/lower coverslip bounds in flattened Z coordinates",
    )
    parser.add_argument("--flattened-template", type=Path,
                        help="Optional flattened TIFF; otherwise output Z is read from surface maps")
    parser.add_argument("--rows", type=int, help="Manual grid-row override (default: infer)")
    parser.add_argument("--columns", type=int, help="Manual grid-column override (default: infer)")
    parser.add_argument("--patch-size-um", type=float, required=True,
                        help="Physical field of view of one acquisition patch in microns")
    parser.add_argument("--background-patch", action="append", type=parse_patch, required=True)
    parser.add_argument("--z-min", type=int, default=20)
    parser.add_argument("--z-max", type=int, default=47)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--noise-sigma", type=float, default=2.5)
    parser.add_argument("--blank-sampling", type=int, default=4)
    parser.add_argument("--blank-envelope-quantile", type=float, default=90.0)
    parser.add_argument("--scale-quantile", type=float, default=20.0)
    parser.add_argument("--minimum-anchor-scale", type=float, default=0.75)
    parser.add_argument("--maximum-anchor-scale", type=float, default=1.25)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Glass-template subtraction strength, normally 0 to 1")
    parser.add_argument("--minimum-contiguous", type=int, default=2)
    parser.add_argument("--mask-downsample", type=int, default=5)
    parser.add_argument("--mask-blank-quantile", type=float, default=93.0)
    parser.add_argument("--mask-neck-opening-iterations", type=int, default=0,
                        help="Low-resolution opening used to detach external glass/bubble regions")
    parser.add_argument("--mask-texture-quantile", type=float, default=0.0,
                        help="Blank texture quantile for excluding smooth external reflections; 0 disables")
    parser.add_argument("--mask-texture-opening-iterations", type=int, default=10)
    parser.add_argument("--comparison-projection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.patch_size_um <= 0:
        parser.error("--patch-size-um must be positive")
    if not 0 <= args.strength <= 1.25:
        parser.error("--strength must be between 0 and 1.25")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = read_geometry(
        args.tiff, args.rows, args.columns, patch_size_um=args.patch_size_um
    )
    volume = open_volume(args.tiff)
    model = load_model(args.model_dir)
    backgrounds = list(dict.fromkeys(args.background_patch))
    for coordinate in backgrounds:
        validate_patch(coordinate, geometry)
    maps = np.load(args.surface_maps)
    full_shift = np.asarray(maps["shift_z"], dtype=np.float32)
    if args.flattened_template is not None:
        with tifffile.TiffFile(args.flattened_template) as tif:
            output_z = int(tif.series[0].shape[0])
    elif "output_z" in maps:
        output_z = int(maps["output_z"])
    else:
        parser.error("Surface maps lack output_z; provide --flattened-template")
    if not (0 <= args.z_min <= args.z_max < output_z):
        parser.error("Invalid projection Z window")

    interface_bounds: dict[tuple[int, int], tuple[int, int]] = {}
    if args.patch_interface_bounds is not None:
        payload = json.loads(args.patch_interface_bounds.read_text())
        if (
            int(payload.get("grid_rows", -1)) != geometry.rows
            or int(payload.get("grid_columns", -1)) != geometry.columns
        ):
            parser.error("Patch-interface bounds do not match the inferred mosaic grid")
        for item in payload.get("patches", []):
            coordinate = (int(item["row"]), int(item["column"]))
            lower = max(args.z_min, int(item["projection_z_min_inclusive"]))
            upper = min(args.z_max, int(item["projection_z_max_inclusive"]))
            if lower >= upper:
                parser.error(
                    f"Patch {coordinate} has no usable depth after intersecting coverslip and safety bounds"
                )
            interface_bounds[coordinate] = (lower, upper)
        if len(interface_bounds) != geometry.rows * geometry.columns:
            parser.error("Patch-interface bounds are incomplete")

    py, px = geometry.patch_shape_yx
    # Learn an amplitude envelope in scanner-local coordinates. This is the
    # critical preservation constraint: tissue patches never define the cap.
    blank_amplitudes = []
    blank_diagnostics: dict[tuple[int, int], dict] = {}
    for coordinate in backgrounds:
        patch = extract_patch(volume, geometry, coordinate).astype(np.float32)
        _, diagnostics = correct_patch(patch, model, strength=0.0)
        blank_diagnostics[coordinate] = diagnostics
        blank_amplitudes.append(np.asarray(diagnostics["amplitude"], dtype=np.float32))
    blank_amplitudes_array = np.stack(blank_amplitudes)
    blank_median_amplitude = np.median(blank_amplitudes_array, axis=0).astype(np.float32)
    blank_upper_amplitude = np.percentile(
        blank_amplitudes_array, args.blank_envelope_quantile, axis=0
    ).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "blank_anchored_amplitude_envelope.npz",
        median=blank_median_amplitude,
        upper=blank_upper_amplitude,
        upper_quantile=np.float32(args.blank_envelope_quantile),
    )

    # Learn residual depth noise after the capped correction and exact surface
    # alignment, using spatial subsamples from blank patches only.
    blank_samples = []
    for coordinate in backgrounds:
        patch = extract_patch(volume, geometry, coordinate).astype(np.float32)
        diagnostics = blank_diagnostics[coordinate]
        corrected, baseline, _, _ = safe_correct(
            patch,
            diagnostics,
            blank_median_amplitude,
            blank_upper_amplitude,
            args.scale_quantile,
            args.minimum_anchor_scale,
            args.maximum_anchor_scale,
            args.strength,
        )
        row, column = coordinate
        shift = full_shift[row * py : (row + 1) * py, column * px : (column + 1) * px]
        flat = flatten_excess(corrected, baseline, shift, output_z)
        sampled = flat[:, :: args.blank_sampling, :: args.blank_sampling]
        blank_samples.append(sampled.reshape(output_z, -1))
    samples = np.concatenate(blank_samples, axis=1)
    blank_median = np.median(samples, axis=1).astype(np.float32)
    blank_sigma = np.maximum(
        1.4826 * np.median(np.abs(samples - blank_median[:, None]), axis=1), 50.0
    ).astype(np.float32)
    del samples, blank_samples

    y_count, x_count = geometry.shape_zyx[1:]
    balanced = np.zeros((y_count, x_count), dtype=np.float32)
    clean_evidence = np.zeros_like(balanced)
    selected_original = np.zeros_like(balanced)
    selected_corrected = np.zeros_like(balanced)
    confidence = np.zeros_like(balanced)
    removal_fraction = np.zeros_like(balanced)
    cap_engagement_map = np.zeros_like(balanced)
    peak_z = np.zeros((y_count, x_count), dtype=np.uint8)
    supported_count = np.zeros((y_count, x_count), dtype=np.uint8)
    patch_rows: list[dict[str, object]] = []
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            coordinate = (row, column)
            patch_z_min, patch_z_max = interface_bounds.get(
                coordinate, (args.z_min, args.z_max)
            )
            z_slice = slice(patch_z_min, patch_z_max + 1)
            threshold = (
                blank_median[z_slice] + args.noise_sigma * blank_sigma[z_slice]
            )[:, None, None]
            patch = extract_patch(volume, geometry, coordinate).astype(np.float32)
            if coordinate in blank_diagnostics:
                diagnostics = blank_diagnostics[coordinate]
            else:
                _, diagnostics = correct_patch(patch, model, strength=0.0)
            corrected, baseline, cap_engagement, anchor_scale = safe_correct(
                patch,
                diagnostics,
                blank_median_amplitude,
                blank_upper_amplitude,
                args.scale_quantile,
                args.minimum_anchor_scale,
                args.maximum_anchor_scale,
                args.strength,
            )
            y0, x0 = row * py, column * px
            shift = full_shift[y0 : y0 + py, x0 : x0 + px]
            original_flat = flatten_excess(patch, baseline, shift, output_z)
            corrected_flat = flatten_excess(corrected, baseline, shift, output_z)
            candidate = corrected_flat[z_slice]
            evidence = np.maximum(candidate - threshold, 0.0)
            positive = evidence > 0
            neighbor_count = convolve1d(
                positive.astype(np.uint8), np.ones(3, dtype=np.uint8), axis=0,
                mode="constant", cval=0,
            )
            supported = positive & (neighbor_count >= args.minimum_contiguous)
            evidence = np.where(supported, evidence, 0.0)
            local_top_k = min(args.top_k, evidence.shape[0])
            indices = np.argpartition(evidence, -local_top_k, axis=0)[-local_top_k:]
            top_evidence = np.take_along_axis(evidence, indices, axis=0)
            top_corrected = np.take_along_axis(candidate, indices, axis=0)
            top_original = np.take_along_axis(original_flat[z_slice], indices, axis=0)
            selected = top_evidence > 0
            count = np.sum(selected, axis=0)
            denominator = np.maximum(count, 1)
            selected_blank_median = blank_median[patch_z_min + indices]
            balanced_patch = np.sum(
                np.where(selected, np.maximum(top_corrected - selected_blank_median, 0.0), 0.0),
                axis=0,
            ) / denominator
            evidence_patch = np.sum(top_evidence, axis=0) / denominator
            original_patch = np.sum(np.where(selected, top_original, 0.0), axis=0) / denominator
            corrected_patch = np.sum(np.where(selected, top_corrected, 0.0), axis=0) / denominator
            removed_patch = np.maximum(original_patch - corrected_patch, 0.0)
            removal_patch = removed_patch / np.maximum(original_patch, 1.0)
            max_zscore = np.max(
                (candidate - blank_median[z_slice, None, None])
                / blank_sigma[z_slice, None, None],
                axis=0,
            )
            confidence_patch = np.clip((max_zscore - args.noise_sigma) / 4.0, 0.0, 1.0)
            peak_patch = patch_z_min + np.argmax(evidence, axis=0)
            peak_patch[count == 0] = 0
            region = np.s_[y0 : y0 + py, x0 : x0 + px]
            balanced[region] = balanced_patch
            clean_evidence[region] = evidence_patch
            selected_original[region] = original_patch
            selected_corrected[region] = corrected_patch
            confidence[region] = confidence_patch
            removal_fraction[region] = removal_patch
            cap_engagement_map[region] = cap_engagement
            peak_z[region] = peak_patch.astype(np.uint8)
            supported_count[region] = np.minimum(count, 255).astype(np.uint8)
            patch_rows.append(
                {
                    "row": row,
                    "column": column,
                    "projection_z_min_inclusive": patch_z_min,
                    "projection_z_max_inclusive": patch_z_max,
                    "is_background_reference": coordinate in backgrounds,
                    "anchor_scale": anchor_scale,
                    "fitted_amplitude_median": float(np.median(diagnostics["amplitude"])),
                    "safe_amplitude_median": float(
                        np.median(np.asarray(diagnostics["amplitude"]) * (1.0 - cap_engagement))
                    ),
                    "cap_engaged_pixel_fraction": float(np.mean(cap_engagement > 1e-4)),
                    "median_cap_reduction_when_engaged": float(
                        np.median(cap_engagement[cap_engagement > 1e-4])
                    ) if np.any(cap_engagement > 1e-4) else 0.0,
                }
            )

    tissue_mask, mask_threshold = largest_tissue_mask(
        clean_evidence,
        selected_original,
        backgrounds,
        geometry.patch_shape_yx,
        args.mask_downsample,
        args.mask_blank_quantile,
        args.mask_neck_opening_iterations,
        args.mask_texture_quantile,
        args.mask_texture_opening_iterations,
    )
    for array in (
        balanced,
        clean_evidence,
        selected_original,
        selected_corrected,
        confidence,
        removal_fraction,
        cap_engagement_map,
        peak_z,
        supported_count,
    ):
        array[~tissue_mask] = 0

    y_um, x_um = geometry.voxel_size_um_zyx[1:]
    primary = np.clip(np.rint(balanced), 0, 65535).astype(np.uint16)
    evidence_uint16 = np.clip(np.rint(clean_evidence), 0, 65535).astype(np.uint16)
    original_uint16 = np.clip(np.rint(selected_original), 0, 65535).astype(np.uint16)
    corrected_uint16 = np.clip(np.rint(selected_corrected), 0, 65535).astype(np.uint16)
    confidence_uint8 = np.clip(np.rint(confidence * 255), 0, 255).astype(np.uint8)
    removal_uint8 = np.clip(np.rint(removal_fraction * 255), 0, 255).astype(np.uint8)
    cap_uint8 = np.clip(np.rint(cap_engagement_map * 255), 0, 255).astype(np.uint8)
    write_imagej(args.output_dir / "tissue_projection_preservation_aware_uint16.tif", primary, y_um, x_um)
    write_imagej(args.output_dir / "tissue_projection_clean_evidence_uint16.tif", evidence_uint16, y_um, x_um)
    write_imagej(args.output_dir / "tissue_projection_selected_original_uint16.tif", original_uint16, y_um, x_um)
    write_imagej(args.output_dir / "tissue_projection_selected_corrected_uint16.tif", corrected_uint16, y_um, x_um)
    write_imagej(args.output_dir / "tissue_mask_uint8.tif", tissue_mask.astype(np.uint8) * 255, y_um, x_um)
    write_imagej(args.output_dir / "tissue_confidence_uint8.tif", confidence_uint8, y_um, x_um)
    write_imagej(args.output_dir / "selected_signal_removal_fraction_uint8.tif", removal_uint8, y_um, x_um)
    write_imagej(args.output_dir / "artifact_cap_engagement_uint8.tif", cap_uint8, y_um, x_um)
    write_imagej(args.output_dir / "tissue_peak_z_uint8.tif", peak_z, y_um, x_um)
    write_imagej(args.output_dir / "tissue_supported_slice_count_uint8.tif", supported_count, y_um, x_um)

    with (args.output_dir / "patch_preservation_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
        writer.writeheader()
        writer.writerows(patch_rows)
    with (args.output_dir / "blank_depth_model.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["z", "blank_median", "blank_mad_sigma", "detection_threshold"])
        writer.writeheader()
        for z in range(output_z):
            writer.writerow(
                {
                    "z": z,
                    "blank_median": float(blank_median[z]),
                    "blank_mad_sigma": float(blank_sigma[z]),
                    "detection_threshold": float(blank_median[z] + args.noise_sigma * blank_sigma[z]),
                }
            )

    inside = tissue_mask & (primary > 0)
    low, high = np.percentile(primary[inside], [1, 99.7])
    plt.imsave(
        args.output_dir / "tissue_projection_preservation_aware_display.png",
        primary,
        cmap="gray",
        vmin=low,
        vmax=high,
    )
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    axes[0, 0].imshow(primary, cmap="gray", vmin=low, vmax=high)
    axes[0, 0].set_title("Primary preservation-aware projection")
    raw_low, raw_high = np.percentile(original_uint16[tissue_mask], [1, 99.7])
    axes[0, 1].imshow(original_uint16, cmap="gray", vmin=raw_low, vmax=raw_high)
    axes[0, 1].set_title("Selected original signal (audit companion)")
    axes[0, 2].imshow(tissue_mask, cmap="gray")
    axes[0, 2].set_title("Tissue mask")
    image = axes[0, 3].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("Tissue confidence")
    fig.colorbar(image, ax=axes[0, 3], shrink=0.7)
    image = axes[1, 0].imshow(removal_fraction, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("Selected-signal subtraction fraction")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.7)
    image = axes[1, 1].imshow(cap_engagement_map, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("Blank-anchor cap engagement")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.7)
    displayed_bounds = list(interface_bounds.values()) or [(args.z_min, args.z_max)]
    image = axes[1, 2].imshow(
        peak_z,
        cmap="turbo",
        vmin=min(value[0] for value in displayed_bounds),
        vmax=max(value[1] for value in displayed_bounds),
    )
    axes[1, 2].set_title("Selected peak Z")
    fig.colorbar(image, ax=axes[1, 2], shrink=0.7)
    image = axes[1, 3].imshow(supported_count, cmap="viridis", vmin=0, vmax=args.top_k)
    axes[1, 3].set_title("Selected contiguous slices")
    fig.colorbar(image, ax=axes[1, 3], shrink=0.7)
    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(args.output_dir / "preservation_aware_projection_qc.png", dpi=180)
    plt.close(fig)

    comparison = None
    if args.comparison_projection:
        old = tifffile.imread(args.comparison_projection).astype(np.float32)
        if old.shape != primary.shape:
            raise ValueError("Comparison projection shape does not match")
        valid = tissue_mask & (old > 0) & (primary > 0)
        comparison = {
            "comparison_projection": str(args.comparison_projection.resolve()),
            "pearson_correlation_inside_shared_mask": float(np.corrcoef(old[valid], primary[valid])[0, 1]),
            "median_preservation_to_previous_ratio": float(
                np.median(primary[valid] / np.maximum(old[valid], 1.0))
            ),
        }

    manifest = {
        "source_tiff": str(args.tiff.resolve()),
        "grid_rows": geometry.rows,
        "grid_columns": geometry.columns,
        "grid_selection_mode": "automatic" if args.rows is None and args.columns is None else "override",
        "patch_size_um": args.patch_size_um,
        "subtraction_strength": args.strength,
        "primary_output": "tissue_projection_preservation_aware_uint16.tif",
        "output_shape_yx": [y_count, x_count],
        "pixel_size_um_yx": [y_um, x_um],
        "background_patches": [list(v) for v in backgrounds],
        "projection_z_window_inclusive": [args.z_min, args.z_max],
        "patch_interface_bounds": (
            str(args.patch_interface_bounds.resolve())
            if args.patch_interface_bounds is not None else None
        ),
        "projection_is_constrained_between_detected_coverslips": bool(interface_bounds),
        "per_patch_projection_z_range": {
            "minimum_start": min(value[0] for value in displayed_bounds),
            "maximum_end": max(value[1] for value in displayed_bounds),
        },
        "top_k": args.top_k,
        "top_k_physical_depth_um": args.top_k * geometry.voxel_size_um_zyx[0],
        "noise_sigma": args.noise_sigma,
        "minimum_contiguous_slices": args.minimum_contiguous,
        "blank_amplitude_envelope_quantile": args.blank_envelope_quantile,
        "patch_anchor_scale_quantile": args.scale_quantile,
        "mask_blank_quantile": args.mask_blank_quantile,
        "mask_neck_opening_iterations": args.mask_neck_opening_iterations,
        "mask_texture_quantile": args.mask_texture_quantile,
        "mask_texture_opening_iterations": args.mask_texture_opening_iterations,
        "mask_threshold": mask_threshold,
        "mask_fraction": float(tissue_mask.mean()),
        "median_selected_signal_removal_fraction": float(np.median(removal_fraction[tissue_mask])),
        "p90_selected_signal_removal_fraction": float(np.percentile(removal_fraction[tissue_mask], 90)),
        "median_cap_engagement": float(np.median(cap_engagement_map[tissue_mask])),
        "comparison": comparison,
        "primary_definition": (
            "Mean of up to five strongest depth-contiguous tissue-supported values after "
            "blank-anchored capped glass subtraction and subtraction of the blank median. "
            "When supplied, detected per-patch coverslip interfaces constrain eligible depths. "
            "The noise threshold is used for selection, not subtracted from retained signal."
        ),
    }
    with (args.output_dir / "preservation_aware_projection_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
