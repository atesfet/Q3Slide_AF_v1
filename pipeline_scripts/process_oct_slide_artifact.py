#!/usr/bin/env python3
"""Learn a glass-slide signature from blank patches and preview/apply correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oct_mpl_cache")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from oct_slide_artifact import (
    build_artifact_model,
    correct_patch,
    extract_patch,
    estimate_background_patches,
    open_volume,
    parse_patch,
    read_geometry,
    save_model,
    validate_patch,
    write_csv,
)


def save_reference_figure(references, coordinates, model, path: Path) -> None:
    fig, axes = plt.subplots(2, len(references) + 1, figsize=(5 * (len(references) + 1), 8),
                             constrained_layout=True, squeeze=False)
    for index, (patch, coordinate) in enumerate(zip(references, coordinates)):
        low, high = np.percentile(patch, [1, 99.8])
        axes[0, index].imshow(patch.max(axis=0), cmap="gray", vmin=low, vmax=high)
        axes[0, index].set_title(f"blank {coordinate}: Z maximum")
        axes[1, index].imshow(patch[:, :, patch.shape[2] // 2], cmap="gray", aspect="auto",
                              vmin=low, vmax=high)
        axes[1, index].set_title("central Z-Y B-scan")
        axes[1, index].set(xlabel="Y pixel", ylabel="Z index")
    axes[0, -1].plot(model.profile, np.arange(len(model.profile)))
    axes[0, -1].invert_yaxis()
    axes[0, -1].set(title="learned normalized profile", xlabel="relative artifact", ylabel="Z")
    axes[0, -1].grid(alpha=0.3)
    radius_um = np.arange(len(model.radial_gain)) * model.voxel_size_um_zyx[2]
    axes[1, -1].plot(radius_um, model.radial_gain, label="amplitude gain")
    second = axes[1, -1].twinx()
    second.plot(radius_um, model.radial_z_offset, color="tab:orange", label="axial offset")
    axes[1, -1].set(title="shared radial fixed pattern", xlabel="radius (µm)",
                    ylabel="relative gain")
    second.set_ylabel("Z offset (slices)", color="tab:orange")
    axes[1, -1].grid(alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_preview(original, corrected, diagnostics, coordinate, path: Path) -> None:
    low, high = np.percentile(original, [1, 99.8])
    difference = original.astype(np.float32) - corrected.astype(np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(original.max(0), cmap="gray", vmin=low, vmax=high)
    axes[0, 0].set_title(f"patch {coordinate}: original Z maximum")
    axes[0, 1].imshow(corrected.max(0), cmap="gray", vmin=low, vmax=high)
    axes[0, 1].set_title("corrected Z maximum")
    axes[0, 2].imshow(difference.max(0), cmap="magma")
    axes[0, 2].set_title("removed component Z maximum")
    middle_x = original.shape[2] // 2
    axes[1, 0].imshow(original[:, :, middle_x], cmap="gray", aspect="auto", vmin=low, vmax=high)
    axes[1, 1].imshow(corrected[:, :, middle_x], cmap="gray", aspect="auto", vmin=low, vmax=high)
    image = axes[1, 2].imshow(diagnostics["surface_z"], cmap="viridis")
    axes[1, 2].set_title(f"estimated slide surface; center={diagnostics['center_z']:.2f}")
    fig.colorbar(image, ax=axes[1, 2], shrink=0.8, label="Z index")
    for ax in axes[1, :2]:
        ax.set(xlabel="Y pixel", ylabel="Z index")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--rows", type=int, help="Manual grid-row override (default: infer)")
    parser.add_argument("--columns", type=int, help="Manual grid-column override (default: infer)")
    parser.add_argument("--patch-size-um", type=float, default=1000.0,
                        help="Physical patch FOV used for grid inference")
    parser.add_argument("--reference-patch", action="append", type=parse_patch,
                        help="Zero-based ROW,COLUMN; repeat. Omit for automatic selection")
    parser.add_argument("--tissue-mask", type=Path,
                        help="Optional prior tissue mask for high-confidence automatic blanks")
    parser.add_argument("--auto-reference-count", type=int, default=10)
    parser.add_argument("--auto-max-tissue-fraction", type=float, default=0.02)
    parser.add_argument("--preview-patch", action="append", type=parse_patch, default=[])
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-corrected-volume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.strength <= 1.25:
        parser.error("--strength must be between 0 and 1.25")

    model_dir = args.output_dir / "model"
    preview_dir = args.output_dir / "previews"
    corrected_dir = args.output_dir / "corrected_volume"
    for directory in (model_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    geometry = read_geometry(args.tiff, args.rows, args.columns, args.patch_size_um)
    volume = open_volume(args.tiff)
    automatic_report = None
    if args.reference_patch:
        reference_patches = list(args.reference_patch)
        selection_mode = "manual"
    else:
        prior_mask = tifffile.imread(args.tissue_mask) if args.tissue_mask else None
        reference_patches, automatic_report = estimate_background_patches(
            volume,
            geometry,
            count=args.auto_reference_count,
            tissue_mask=prior_mask,
            maximum_tissue_fraction=args.auto_max_tissue_fraction,
        )
        selection_mode = "automatic-mask-refined" if prior_mask is not None else "automatic-bootstrap"
        write_csv(automatic_report, args.output_dir / "automatic_background_patch_report.csv")
    for coordinate in [*reference_patches, *args.preview_patch]:
        validate_patch(coordinate, geometry)
    references = [extract_patch(volume, geometry, p) for p in reference_patches]
    model = build_artifact_model(
        references, reference_patches, geometry.voxel_size_um_zyx
    )
    save_model(model, model_dir)
    save_reference_figure(
        references, reference_patches, model, model_dir / "reference_characterization.png"
    )

    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "source_tiff": str(args.tiff.resolve()),
                "grid_rows": geometry.rows,
                "grid_columns": geometry.columns,
                "patch_size_um": args.patch_size_um,
                "grid_selection_mode": (
                    "automatic" if args.rows is None and args.columns is None else
                    "manual" if args.rows is not None and args.columns is not None else
                    "mixed"
                ),
                "coordinate_convention": "zero-based (row,column)",
                "reference_patches": [list(v) for v in reference_patches],
                "reference_selection_mode": selection_mode,
                "requested_preview_patches": [list(v) for v in args.preview_patch],
                "subtraction_strength": args.strength,
                "full_volume_requested": args.write_corrected_volume,
            },
            handle,
            indent=2,
        )

    preview_coordinates = list(dict.fromkeys([*reference_patches, *args.preview_patch]))
    preview_metrics = []
    for coordinate in preview_coordinates:
        original = extract_patch(volume, geometry, coordinate)
        corrected, diagnostics = correct_patch(original, model, strength=args.strength)
        stem = f"patch_r{coordinate[0]:02d}_c{coordinate[1]:02d}"
        save_preview(original, corrected, diagnostics, coordinate, preview_dir / f"{stem}_preview.png")
        tifffile.imwrite(
            preview_dir / f"{stem}_corrected.tif",
            corrected,
            imagej=True,
            metadata={"axes": "ZYX", "unit": "micron", "spacing": geometry.voxel_size_um_zyx[0]},
            resolution=(1.0 / geometry.voxel_size_um_zyx[2], 1.0 / geometry.voxel_size_um_zyx[1]),
        )
        np.savez_compressed(
            preview_dir / f"{stem}_diagnostics.npz",
            center_z=np.float32(diagnostics["center_z"]),
            surface_z=diagnostics["surface_z"].astype(np.float32),
            baseline=diagnostics["baseline"].astype(np.float32),
            amplitude=diagnostics["amplitude"].astype(np.float32),
        )
        original_profile = np.median(original.astype(np.float32), axis=(1, 2))
        corrected_profile = np.median(corrected.astype(np.float32), axis=(1, 2))
        original_baseline = float(np.quantile(original_profile, model.baseline_quantile))
        corrected_baseline = float(np.quantile(corrected_profile, model.baseline_quantile))
        original_peak = float(original_profile.max() - original_baseline)
        corrected_peak = float(corrected_profile.max() - corrected_baseline)
        preview_metrics.append(
            {
                "row": coordinate[0],
                "column": coordinate[1],
                "is_reference": coordinate in reference_patches,
                "estimated_center_z": float(diagnostics["center_z"]),
                "original_median_profile_peak_excess": original_peak,
                "corrected_median_profile_peak_excess": corrected_peak,
                "peak_excess_reduction_percent": 100.0 * (1.0 - corrected_peak / max(original_peak, 1e-6)),
                "mean_removed_stored_units": float(
                    original.astype(np.float32).mean() - corrected.astype(np.float32).mean()
                ),
            }
        )
    write_csv(preview_metrics, preview_dir / "preview_metrics.csv")

    if args.write_corrected_volume:
        corrected_dir.mkdir(parents=True, exist_ok=True)
        output_path = corrected_dir / f"{args.tiff.stem}_glass_corrected.tif"
        output = np.memmap(
            corrected_dir / f".{args.tiff.stem}_glass_corrected.tmp",
            mode="w+", dtype=np.uint16, shape=geometry.shape_zyx
        )
        manifest = []
        for row in range(geometry.rows):
            for column in range(geometry.columns):
                coordinate = (row, column)
                corrected, diagnostics = correct_patch(
                    extract_patch(volume, geometry, coordinate), model, strength=args.strength
                )
                patch_y, patch_x = geometry.patch_shape_yx
                output[:, row * patch_y : (row + 1) * patch_y,
                       column * patch_x : (column + 1) * patch_x] = corrected
                manifest.append({"row": row, "column": column,
                                 "estimated_center_z": float(diagnostics["center_z"])})
        output.flush()
        tifffile.imwrite(
            output_path,
            output,
            # Prefer conventional TIFF when safely below the 4 GiB offset limit;
            # ImageJ BigTIFF is a nonstandard extension and less portable.
            bigtiff=output.nbytes > 3_900_000_000,
            imagej=True,
            metadata={"axes": "ZYX", "unit": "micron", "spacing": geometry.voxel_size_um_zyx[0]},
            resolution=(1.0 / geometry.voxel_size_um_zyx[2], 1.0 / geometry.voxel_size_um_zyx[1]),
        )
        del output
        (corrected_dir / f".{args.tiff.stem}_glass_corrected.tmp").unlink()
        with (corrected_dir / "correction_manifest.json").open("w") as handle:
            json.dump({"source": str(args.tiff.resolve()), "output": str(output_path.resolve()),
                       "strength": args.strength, "patches": manifest}, handle, indent=2)

    print(f"Wrote artifact model to {model_dir}")
    print(f"Wrote {len(preview_coordinates)} correction preview(s) to {preview_dir}")


if __name__ == "__main__":
    main()
