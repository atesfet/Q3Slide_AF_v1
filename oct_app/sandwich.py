from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import GeometrySummary, UserFacingError


def derive_patch_interfaces(
    blocks_csv: Path,
    geometry: GeometrySummary,
    output_dir: Path,
    minimum_valid_blocks: int = 3,
) -> dict:
    """Aggregate block-level interface measurements into one sandwich per patch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[int, int], list[dict]] = {
        (row, column): []
        for row in range(geometry.rows)
        for column in range(geometry.columns)
    }
    with blocks_csv.open(newline="") as handle:
        for item in csv.DictReader(handle):
            row = min(
                geometry.rows - 1,
                max(0, int(float(item["center_y_um"]) // geometry.patch_fov_um)),
            )
            column = min(
                geometry.columns - 1,
                max(0, int(float(item["center_x_um"]) // geometry.patch_fov_um)),
            )
            groups[(row, column)].append(item)

    upper = np.full((geometry.rows, geometry.columns), np.nan, np.float32)
    lower = np.full_like(upper, np.nan)
    quality = np.full_like(upper, np.nan)
    valid_fraction = np.zeros_like(upper)
    valid_counts = np.zeros_like(upper, dtype=np.int32)
    total_counts = np.zeros_like(upper, dtype=np.int32)
    sources = np.full((geometry.rows, geometry.columns), "measured", dtype=object)

    for (row, column), items in groups.items():
        total_counts[row, column] = len(items)
        valid = [item for item in items if item["valid"] == "1"]
        valid_counts[row, column] = len(valid)
        valid_fraction[row, column] = len(valid) / max(len(items), 1)
        if len(valid) >= minimum_valid_blocks:
            upper[row, column] = np.median(
                [float(item["first_interface_z_um"]) for item in valid]
            )
            lower[row, column] = np.median(
                [float(item["second_interface_z_um"]) for item in valid]
            )
            quality[row, column] = np.median([float(item["quality"]) for item in valid])

    measured = np.isfinite(upper) & np.isfinite(lower) & (lower > upper)
    if measured.sum() < 2:
        raise UserFacingError(
            "Fewer than two patches contained reliable upper and lower coverslip detections. "
            "The volume cannot yet be constrained to a tissue sandwich."
        )

    measured_positions = np.argwhere(measured)
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            if measured[row, column]:
                continue
            distances = np.sum((measured_positions - np.array([row, column])) ** 2, axis=1)
            nearest_row, nearest_column = measured_positions[int(np.argmin(distances))]
            upper[row, column] = upper[nearest_row, nearest_column]
            lower[row, column] = lower[nearest_row, nearest_column]
            quality[row, column] = quality[nearest_row, nearest_column]
            sources[row, column] = "nearest-patch fallback"

    separation = lower - upper
    records = []
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            records.append(
                {
                    "row": row,
                    "column": column,
                    "upper_interface_z_um": float(upper[row, column]),
                    "lower_interface_z_um": float(lower[row, column]),
                    "sandwich_depth_um": float(separation[row, column]),
                    "median_detection_quality": float(quality[row, column]),
                    "valid_block_count": int(valid_counts[row, column]),
                    "total_block_count": int(total_counts[row, column]),
                    "valid_block_fraction": float(valid_fraction[row, column]),
                    "source": str(sources[row, column]),
                }
            )

    csv_path = output_dir / "patch_interface_report.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "coordinate_convention": "zero-based row,column",
        "grid_rows": geometry.rows,
        "grid_columns": geometry.columns,
        "patch_fov_um": geometry.patch_fov_um,
        "minimum_valid_blocks_per_patch": minimum_valid_blocks,
        "measured_patch_count": int(measured.sum()),
        "fallback_patch_count": int((~measured).sum()),
        "median_sandwich_depth_um": float(np.median(separation)),
        "p05_sandwich_depth_um": float(np.percentile(separation, 5)),
        "p95_sandwich_depth_um": float(np.percentile(separation, 95)),
        "patches": records,
    }
    json_path = output_dir / "patch_interface_bounds.json"
    json_path.write_text(json.dumps(payload, indent=2))

    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    panels = [
        (upper, "Upper interface Z (µm)", "viridis"),
        (lower, "Lower interface Z (µm)", "viridis"),
        (separation, "Tissue sandwich depth (µm)", "magma"),
        (valid_fraction, "Valid local detection fraction", "cividis"),
    ]
    for axis, (array, title, cmap) in zip(axes, panels):
        image = axis.imshow(array, cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("column")
        axis.set_ylabel("row")
        figure.colorbar(image, ax=axis, shrink=0.75)
        for row in range(geometry.rows):
            for column in range(geometry.columns):
                if sources[row, column] != "measured":
                    axis.plot(column, row, marker="x", color="white", markersize=5)
    figure.suptitle("Per-patch coverslip sandwich; white × indicates nearest-patch fallback")
    figure.savefig(output_dir / "patch_interface_qc.png", dpi=180)
    plt.close(figure)
    return payload


def convert_to_flattened_bounds(
    patch_payload: dict,
    geometry: GeometrySummary,
    surface_maps: Path,
    output_path: Path,
    margin_um: float = 0.0,
) -> dict:
    """Convert raw interface depths into the flattened projection coordinate system."""
    if margin_um < 0:
        raise UserFacingError("Interface margin cannot be negative.")
    maps = np.load(surface_maps)
    shift = np.asarray(maps["shift_z"], dtype=np.float32)
    output_z = int(maps["output_z"])
    z_um = float(geometry.voxel_size_um_zyx[0])
    margin_z = margin_um / z_um
    py, px = geometry.patch_shape_yx
    records = []
    for item in patch_payload["patches"]:
        row, column = int(item["row"]), int(item["column"])
        local_shift = shift[row * py : (row + 1) * py, column * px : (column + 1) * px]
        shift_median = float(np.median(local_shift))
        upper_raw = float(item["upper_interface_z_um"]) / z_um
        lower_raw = float(item["lower_interface_z_um"]) / z_um
        upper_flat = max(0, int(np.floor(upper_raw + shift_median + margin_z)))
        lower_flat = min(output_z - 1, int(np.ceil(lower_raw + shift_median - margin_z)))
        if lower_flat <= upper_flat:
            raise UserFacingError(
                f"Detected coverslip bounds collapsed in patch ({row},{column}). "
                "Reduce the interface margin or review the interface detections."
            )
        records.append(
            {
                **item,
                "median_flattening_shift_z": shift_median,
                "projection_z_min_inclusive": upper_flat,
                "projection_z_max_inclusive": lower_flat,
            }
        )
    payload = {
        "grid_rows": geometry.rows,
        "grid_columns": geometry.columns,
        "output_z": output_z,
        "axial_spacing_um": z_um,
        "interface_margin_um": margin_um,
        "method": "Per-patch raw interface depths shifted into the flattened slide-relative coordinate system",
        "patches": records,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return payload
