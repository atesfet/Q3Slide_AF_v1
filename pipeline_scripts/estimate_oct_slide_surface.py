#!/usr/bin/env python3
"""Estimate slide-surface flattening maps without writing a flattened volume."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oct_mpl_cache")
import matplotlib.pyplot as plt
import numpy as np

from oct_slide_artifact import (
    estimate_patch_center,
    estimate_surface_map,
    extract_patch,
    load_model,
    open_volume,
    read_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, help="Manual grid-row override (default: infer)")
    parser.add_argument("--columns", type=int, help="Manual grid-column override (default: infer)")
    parser.add_argument("--patch-size-um", type=float, default=1000.0,
                        help="Physical patch FOV used for grid inference")
    parser.add_argument("--padding-z", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = read_geometry(args.tiff, args.rows, args.columns, args.patch_size_um)
    volume = open_volume(args.tiff)
    model = load_model(args.model_dir)
    surface = np.empty(geometry.shape_zyx[1:], dtype=np.float32)
    py, px = geometry.patch_shape_yx
    records = []
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            patch = extract_patch(volume, geometry, (row, column))
            center = estimate_patch_center(patch, model)
            local = estimate_surface_map(patch, model, patch_center=center)
            y0, x0 = row * py, column * px
            surface[y0:y0 + py, x0:x0 + px] = local
            records.append({
                "row": row, "column": column, "center_z": center,
                "surface_min_z": float(local.min()),
                "surface_median_z": float(np.median(local)),
                "surface_max_z": float(local.max()),
            })

    minimum = float(surface.min())
    maximum = float(surface.max())
    target_z = int(np.ceil(maximum)) + args.padding_z
    output_z = int(np.ceil((geometry.shape_zyx[0] - 1) + target_z - minimum)) + 1 + args.padding_z
    shift = (target_z - surface).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "slide_surface_and_shift_maps.npz",
        surface_z=surface, shift_z=shift, target_z=np.int32(target_z),
        output_z=np.int32(output_z),
        axial_spacing_um=np.float64(geometry.voxel_size_um_zyx[0]),
    )
    with (args.output_dir / "patch_surface_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    centers = np.array([v["center_z"] for v in records]).reshape(geometry.rows, geometry.columns)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    im = axes[0].imshow(centers, cmap="viridis"); axes[0].set_title("Patch slide centers")
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    im = axes[1].imshow(surface[::10, ::10], cmap="viridis"); axes[1].set_title("Estimated slide surface")
    fig.colorbar(im, ax=axes[1], shrink=0.75)
    im = axes[2].imshow(shift[::10, ::10], cmap="coolwarm"); axes[2].set_title(f"Shift to Z={target_z}")
    fig.colorbar(im, ax=axes[2], shrink=0.75)
    for axis in axes: axis.axis("off")
    fig.savefig(args.output_dir / "surface_estimation_qc.png", dpi=180)
    plt.close(fig)

    manifest = {
        "source_tiff": str(args.tiff.resolve()),
        "grid_rows": geometry.rows, "grid_columns": geometry.columns,
        "patch_shape_yx": list(geometry.patch_shape_yx),
        "patch_size_um": args.patch_size_um,
        "voxel_size_um_zyx": list(geometry.voxel_size_um_zyx),
        "surface_min_z": minimum, "surface_max_z": maximum,
        "target_slide_z": target_z, "output_z": output_z,
        "applied_shift_min_z": float(shift.min()),
        "applied_shift_max_z": float(shift.max()),
        "maps_only": True,
    }
    with (args.output_dir / "surface_estimation_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
