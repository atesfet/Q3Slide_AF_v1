from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from pipeline_scripts.tiff_calibration import read_z_spacing_um


APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = APP_ROOT / "pipeline_scripts"


class UserFacingError(RuntimeError):
    """An error that can be shown directly in the web interface."""


@dataclass(frozen=True)
class GeometrySummary:
    shape_zyx: tuple[int, int, int]
    dtype: str
    axes: str
    voxel_size_um_zyx: tuple[float, float, float]
    rows: int
    columns: int
    patch_shape_yx: tuple[int, int]
    patch_fov_um: float
    suggested_fov_um: float
    suggestion_reason: str

    def as_dict(self) -> dict:
        return {
            "shape_zyx": list(self.shape_zyx),
            "dtype": self.dtype,
            "axes": self.axes,
            "voxel_size_um_zyx": list(self.voxel_size_um_zyx),
            "rows": self.rows,
            "columns": self.columns,
            "patch_shape_yx": list(self.patch_shape_yx),
            "patch_fov_um": self.patch_fov_um,
            "suggested_fov_um": self.suggested_fov_um,
            "suggestion_reason": self.suggestion_reason,
        }


def _rational_to_float(value) -> float:
    if isinstance(value, tuple):
        return float(value[0]) / float(value[1])
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    return float(value)


def suggest_fov(path: Path, fallback: float = 700.0) -> tuple[float, str]:
    name = path.name.lower().replace(" ", "")
    micron_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:um|µm)(?:fov)?", name)
    # Section-thickness names such as 10um_H&E are not patch FOV values.
    for matched in reversed(micron_matches):
        value = float(matched)
        if value >= 100:
            return value, "Read from the filename"
    mm_match = re.search(r"(\d+(?:\.\d+)?)\s*mm(?:fov)?", name)
    if mm_match:
        return float(mm_match.group(1)) * 1000.0, "Read from the filename"
    return float(fallback), "Default value; confirm against the scanner acquisition"


def read_tiff_spacing(path: Path) -> tuple[tuple[int, int, int], str, str, tuple[float, float, float]]:
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            if len(series.shape) != 3:
                raise UserFacingError(
                    f"Expected a 3D Z-Y-X TIFF, but this file has shape {series.shape}."
                )
            axes = series.axes
            if axes != "ZYX":
                raise UserFacingError(
                    f"Expected TIFF axes ZYX, but the file reports {axes}. Export the OCT volume as a Z stack."
                )
            metadata = tif.imagej_metadata or {}
            try:
                z_um = read_z_spacing_um(metadata)
            except ValueError as exc:
                raise UserFacingError(str(exc)) from exc
            page = tif.pages[0]
            x_tag = page.tags.get("XResolution")
            y_tag = page.tags.get("YResolution")
            if x_tag is None or y_tag is None:
                raise UserFacingError(
                    "The TIFF does not contain lateral pixel spacing. Re-export it with X/Y resolution metadata."
                )
            x_ppu = _rational_to_float(x_tag.value)
            y_ppu = _rational_to_float(y_tag.value)
            if x_ppu <= 0 or y_ppu <= 0:
                raise UserFacingError("The TIFF contains invalid X/Y resolution metadata.")
            return (
                tuple(int(v) for v in series.shape),
                str(series.dtype),
                axes,
                (z_um, 1.0 / y_ppu, 1.0 / x_ppu),
            )
    except UserFacingError:
        raise
    except Exception as exc:
        raise UserFacingError(f"The TIFF could not be opened: {exc}") from exc


def inspect_geometry(
    path: Path,
    fov_um: float,
    rows: int | None = None,
    columns: int | None = None,
) -> GeometrySummary:
    if not path.is_file():
        raise UserFacingError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise UserFacingError("Select a .tif or .tiff OCT volume.")
    if not np.isfinite(fov_um) or fov_um <= 0:
        raise UserFacingError("Patch FOV is required and must be greater than zero microns.")
    shape, dtype, axes, spacing = read_tiff_spacing(path)
    _, ny, nx = shape
    _, y_um, x_um = spacing

    def infer(pixel_count: int, pixel_um: float, supplied: int | None, axis: str) -> int:
        if supplied is not None:
            if supplied <= 0:
                raise UserFacingError(f"Manual {axis} count must be positive.")
            count = int(supplied)
        else:
            estimate = pixel_count * pixel_um / fov_um
            count = int(round(estimate))
            if count <= 0 or abs(estimate - count) > 0.05:
                raise UserFacingError(
                    f"The {axis} extent implies {estimate:.3f} patches at {fov_um:g} µm FOV. "
                    "Confirm the patch FOV or enter the row and column counts manually."
                )
        if pixel_count % count:
            raise UserFacingError(
                f"The image has {pixel_count} pixels along {axis}, which is not divisible by {count} patches. "
                "Confirm FOV and grid dimensions."
            )
        return count

    resolved_rows = infer(ny, y_um, rows, "row")
    resolved_columns = infer(nx, x_um, columns, "column")
    suggested, reason = suggest_fov(path)
    return GeometrySummary(
        shape_zyx=shape,
        dtype=dtype,
        axes=axes,
        voxel_size_um_zyx=spacing,
        rows=resolved_rows,
        columns=resolved_columns,
        patch_shape_yx=(ny // resolved_rows, nx // resolved_columns),
        patch_fov_um=float(fov_um),
        suggested_fov_um=suggested,
        suggestion_reason=reason,
    )


def parse_background_patches(text: str) -> list[tuple[int, int]]:
    text = text.strip()
    if not text:
        return []
    found = re.findall(r"(\d+)\s*[, :]\s*(\d+)", text)
    leftovers = re.sub(r"(\d+)\s*[, :]\s*(\d+)", "", text)
    leftovers = re.sub(r"[;\s]+", "", leftovers)
    if leftovers or not found:
        raise UserFacingError(
            "Background patches must use row,column pairs, for example: 0,5; 0,6; 1,9"
        )
    return list(dict.fromkeys((int(r), int(c)) for r, c in found))


def explain_failure(message: str) -> str:
    lowered = message.lower()
    if 'memory-mapp' in lowered or 'memorymap' in lowered:
        return 'This TIFF cannot be memory-mapped. Re-export it as an uncompressed Z stack and try again.'
    if "cannot reliably infer" in lowered or "physical extent implies" in lowered:
        return "The FOV does not produce an integer patch grid. Confirm the scanner patch FOV or enter rows and columns manually."
    if "could not identify two" in lowered or "no interface pair" in lowered:
        return "Two coverslip interfaces were not detected. The image may not contain both interfaces, or the minimum separation/search settings may need adjustment."
    if "no tissue component" in lowered or "no reliable" in lowered and "tissue" in lowered:
        return "No reliable tissue region was found. Check the background references and expand the slide-relative depth search range."
    if "background" in lowered and ("patch" in lowered or "reference" in lowered):
        return "Reliable blank patches could not be established. Provide at least two known background patch coordinates in manual mode."
    if "memoryerror" in lowered or "cannot allocate memory" in lowered:
        return "The volume is larger than available memory. Close other applications or run the analysis on a workstation with more RAM."
    if "axes" in lowered or "expected one z,y,x" in lowered or "expected zyx" in lowered:
        return "The input is not stored as a Z-Y-X OCT stack. Re-export it as a 3D TIFF volume."
    return "The analysis stopped before completion. Review the final log lines below; they preserve the original technical error for reporting."


def is_missing_coverslip_pair(message: str) -> bool:
    """Only a missing/unreliable interface pair permits unconstrained preprocessing."""
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "could not identify two distinct reflective interfaces",
            "no interface pair met the requested minimum separation",
            "no reliable local interface pairs remained",
            "fewer than two patches contained reliable upper and lower coverslip detections",
        )
    )


def _append_grid_args(command: list[str], rows: int | None, columns: int | None) -> None:
    if rows is not None:
        command += ["--rows", str(rows)]
    if columns is not None:
        command += ["--columns", str(columns)]


def validate_reference_patches(reviewed: list, geometry: GeometrySummary) -> list[tuple[int, int]]:
    if not isinstance(reviewed, list) or any(
        not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p)
        for p in reviewed
    ):
        raise UserFacingError('Reference tiles must be integer row,column pairs.')
    patches = list(dict.fromkeys(tuple(p) for p in reviewed))
    if len(patches) < 2:
        raise UserFacingError('Select at least two no-tissue reference tiles before preprocessing.')
    for row, column in patches:
        if not (0 <= row < geometry.rows and 0 <= column < geometry.columns):
            raise UserFacingError(f'Reference tile {row},{column} is outside this image grid.')
    return patches


def run_pipeline(
    config: dict,
    run_dir: Path,
    run_command: Callable[[list[str], str, int], None],
    update: Callable[[str, int], None],
) -> dict:
    image_path = Path(config["image_path"]).expanduser().resolve()
    fov_um = float(config["fov_um"])
    auto_mode = bool(config.get("auto_mode", True))
    rows = None if auto_mode or not config.get("rows") else int(config["rows"])
    columns = None if auto_mode or not config.get("columns") else int(config["columns"])
    geometry = inspect_geometry(image_path, fov_um, rows, columns)
    action = config.get("action", "both")
    if action not in {"preprocess", "spacing", "both"}:
        raise UserFacingError("Choose preprocessing, coverslip spacing, or both analyses.")
    if config.get('reference_patches') is not None:
        validate_reference_patches(config['reference_patches'], geometry)

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_request.json").write_text(json.dumps(config, indent=2))
    (run_dir / "input_geometry.json").write_text(json.dumps(geometry.as_dict(), indent=2))
    result: dict = {"geometry": geometry.as_dict(), "action": action, "files": [], "errors": []}

    # The two physical interfaces are measured before any correction.  Their
    # per-patch bounds become the admissible tissue compartment downstream.
    update("Detecting upper and lower coverslip interfaces", 8)
    spacing_dir = run_dir / "00_coverslip_spacing"
    spacing_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "measure_coverslip_spacing.py"),
        str(image_path),
        "--output-dir", str(spacing_dir),
        "--block-size-um", str(float(config.get("block_size_um", 100.0))),
        "--search-radius-um", str(float(config.get("search_radius_um", 24.0))),
        "--minimum-separation-um", str(float(config.get("minimum_separation_um", 20.0))),
        "--minimum-quality", str(float(config.get("minimum_quality", 2.0))),
    ]
    refractive_index = config.get("refractive_index")
    if refractive_index not in (None, ""):
        spacing_cmd += ["--refractive-index", str(float(refractive_index))]
    # Local import avoids a module-initialization cycle: sandwich uses the
    # geometry and friendly-error types defined above.
    from .sandwich import derive_patch_interfaces, convert_to_flattened_bounds
    patch_payload = None
    try:
        run_command(spacing_cmd, "coverslip-spacing", 23)
        spacing_summary = json.loads((spacing_dir / "coverslip_spacing_summary.json").read_text())
        result["coverslip_spacing"] = spacing_summary
        update("Summarizing the tissue sandwich for each patch", 25)
        patch_payload = derive_patch_interfaces(
            spacing_dir / "coverslip_spacing_blocks.csv", geometry, spacing_dir
        )
        result["patch_sandwich"] = patch_payload
        result["files"] += [
            {"label": "Coverslip QC", "path": str((spacing_dir / "coverslip_spacing_qc.png").relative_to(run_dir)), "kind": "image"},
            {"label": "Per-patch sandwich QC", "path": str((spacing_dir / "patch_interface_qc.png").relative_to(run_dir)), "kind": "image"},
            {"label": "Per-patch interface report", "path": str((spacing_dir / "patch_interface_report.csv").relative_to(run_dir)), "kind": "download"},
            {"label": "Patch interface bounds", "path": str((spacing_dir / "patch_interface_bounds.json").relative_to(run_dir)), "kind": "download"},
            {"label": "Spacing summary", "path": str((spacing_dir / "coverslip_spacing_summary.json").relative_to(run_dir)), "kind": "download"},
            {"label": "Local measurements", "path": str((spacing_dir / "coverslip_spacing_blocks.csv").relative_to(run_dir)), "kind": "download"},
            {"label": "Separation map", "path": str((spacing_dir / "coverslip_separation_um.tif").relative_to(run_dir)), "kind": "download"},
        ]
    except (RuntimeError, UserFacingError) as exc:
        if action == "spacing" or not is_missing_coverslip_pair(str(exc)):
            raise
        result["errors"].append({
            "code": "COVERSLIP_PAIR_NOT_DETECTED",
            "message": (
                "Reliable upper/lower coverslip bounds could not be established for the patch grid. "
                "Preprocessing continued without per-patch coverslip depth bounds; "
                "the projection and any available separation measurements need visual review."
            ),
            "technical_detail": str(exc),
        })
        update("Coverslip detection error; continuing preprocessing", 25)

    if action == "spacing":
        result["files"].append(
            {"label": "Complete run summary", "path": "result_summary.json", "kind": "download"}
        )
        (run_dir / "result_summary.json").write_text(json.dumps(result, indent=2))
        update("Complete", 100)
        return result

    if action in {"preprocess", "both"}:
        update("Learning the glass artifact from blank patches", 28)
        artifact_dir = run_dir / "01_artifact_model"
        model_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_oct_slide_artifact.py"),
            str(image_path),
            "--patch-size-um", str(fov_um),
            "--auto-reference-count", str(int(config.get("auto_reference_count", 8))),
            "--strength", str(float(config.get("strength", 1.0))),
            "--output-dir", str(artifact_dir),
        ]
        _append_grid_args(model_cmd, rows, columns)
        manual_backgrounds = parse_background_patches(config.get("background_patches", ""))
        reviewed = config.get('reference_patches')
        if reviewed is not None:
            manual_backgrounds = validate_reference_patches(reviewed, geometry)
        if reviewed is not None or not auto_mode:
            if len(manual_backgrounds) < 2:
                raise UserFacingError(
                    "Select at least two no-tissue reference tiles before preprocessing."
                )
            for row, column in manual_backgrounds:
                if not (0 <= row < geometry.rows and 0 <= column < geometry.columns):
                    raise UserFacingError(f'Reference tile {row},{column} is outside this image grid.')
                model_cmd += ["--reference-patch", f"{row},{column}"]
        if reviewed is not None:
            (run_dir / 'reference_selection.json').write_text(json.dumps({
                'selection_mode': 'reviewed-preview',
                'reference_patches': [list(p) for p in manual_backgrounds],
                'preview_id': config.get('reference_preview_id'),
                'preview_provenance': config.get('reference_preview_provenance'),
            }, indent=2))
            preview_id = str(config.get('reference_preview_id', ''))
            if re.fullmatch(r'[0-9a-f]{12}', preview_id):
                preview_dir = APP_ROOT / '.runtime' / 'previews' / preview_id
                review_dir = run_dir / '00_reference_review'
                review_dir.mkdir(exist_ok=True)
                for name in ['raw_projection.png', 'contrast_projection.png', 'preview_summary.json']:
                    if (preview_dir / name).is_file():
                        shutil.copy2(preview_dir / name, review_dir / name)
                if (review_dir / 'raw_projection.png').is_file():
                    from .preview import save_reference_overlay
                    save_reference_overlay(review_dir/'raw_projection.png', review_dir/'reviewed_reference_tiles.png',
                        geometry.rows, geometry.columns, manual_backgrounds)
                    result['files'].append({'label': 'Reviewed reference mosaic',
                        'path': '00_reference_review/reviewed_reference_tiles.png', 'kind': 'image'})
        run_command(model_cmd, "artifact-model", 45)

        model_config = json.loads((artifact_dir / "run_config.json").read_text())
        backgrounds = [tuple(v) for v in model_config["reference_patches"]]
        if len(backgrounds) < 2:
            raise UserFacingError("Fewer than two blank reference patches were selected.")
        result["background_patches"] = [list(v) for v in backgrounds]
        result["background_selection_mode"] = model_config["reference_selection_mode"]
        if reviewed is not None:
            result['background_selection_mode'] = 'reviewed-preview'
            result['files'].append({'label': 'Reviewed reference tiles', 'path': 'reference_selection.json', 'kind': 'download'})

        update("Estimating the slide surface and Z offsets", 48)
        surface_dir = run_dir / "02_slide_surface"
        surface_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "estimate_oct_slide_surface.py"),
            str(image_path),
            "--model-dir", str(artifact_dir / "model"),
            "--patch-size-um", str(fov_um),
            "--output-dir", str(surface_dir),
        ]
        _append_grid_args(surface_cmd, rows, columns)
        run_command(surface_cmd, "slide-surface", 66)
        surface_manifest = json.loads((surface_dir / "surface_estimation_manifest.json").read_text())
        target_z = int(surface_manifest["target_slide_z"])
        output_z = int(surface_manifest["output_z"])
        z_um = float(geometry.voxel_size_um_zyx[0])
        interface_margin_um = float(config.get("interface_margin_um", 0.0))
        flattened_bounds_path = None
        if patch_payload is not None:
            flattened_bounds_path = surface_dir / "patch_interface_bounds_flattened.json"
            flattened_bounds = convert_to_flattened_bounds(
                patch_payload,
                geometry,
                surface_dir / "slide_surface_and_shift_maps.npz",
                flattened_bounds_path,
                margin_um=interface_margin_um,
            )
            result["flattened_patch_bounds"] = flattened_bounds

        update("Selecting tissue-supported slices", 70)
        projection_dir = run_dir / "03_preprocessed_projection"
        projection_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_preservation_aware_projection.py"),
            str(image_path),
            "--model-dir", str(artifact_dir / "model"),
            "--surface-maps", str(surface_dir / "slide_surface_and_shift_maps.npz"),
            "--patch-size-um", str(fov_um),
            "--z-min", "0",
            "--z-max", str(output_z - 1),
            "--top-k", str(int(config.get("top_k", 5))),
            "--noise-sigma", str(float(config.get("noise_sigma", 2.5))),
            "--strength", str(float(config.get("strength", 1.0))),
            "--output-dir", str(projection_dir),
        ]
        if flattened_bounds_path is not None:
            projection_cmd += ["--patch-interface-bounds", str(flattened_bounds_path)]
        _append_grid_args(projection_cmd, rows, columns)
        for row, column in backgrounds:
            projection_cmd += ["--background-patch", f"{row},{column}"]
        run_command(projection_cmd, "tissue-projection", 94)
        primary = projection_dir / "tissue_projection_preservation_aware_uint16.tif"
        deliverable = run_dir / f"{image_path.stem}_preprocessed.tif"
        shutil.copy2(primary, deliverable)
        projection_manifest = json.loads(
            (projection_dir / "preservation_aware_projection_manifest.json").read_text()
        )
        result["preprocessing"] = projection_manifest
        result["files"] += [
            {"label": "Preprocessed TIFF", "path": deliverable.name, "kind": "download"},
            {"label": "Projection preview", "path": str((projection_dir / "tissue_projection_preservation_aware_display.png").relative_to(run_dir)), "kind": "image"},
            {"label": "Selected original signal TIFF", "path": str((projection_dir / "tissue_projection_selected_original_uint16.tif").relative_to(run_dir)), "kind": "download"},
            {"label": "Selected corrected signal TIFF", "path": str((projection_dir / "tissue_projection_selected_corrected_uint16.tif").relative_to(run_dir)), "kind": "download"},
            {"label": "Selected signal removal fraction", "path": str((projection_dir / "selected_signal_removal_fraction_uint8.tif").relative_to(run_dir)), "kind": "download"},
            {"label": "Preservation QC", "path": str((projection_dir / "preservation_aware_projection_qc.png").relative_to(run_dir)), "kind": "image"},
            {"label": "Surface QC", "path": str((surface_dir / "surface_estimation_qc.png").relative_to(run_dir)), "kind": "image"},
            {"label": "Background characterization", "path": str((artifact_dir / "model/reference_characterization.png").relative_to(run_dir)), "kind": "image"},
        ]
        if flattened_bounds_path is not None:
            result["files"].append(
                {"label": "Flattened patch bounds", "path": str(flattened_bounds_path.relative_to(run_dir)), "kind": "download"}
            )

    result["files"].append(
        {"label": "Complete run summary", "path": "result_summary.json", "kind": "download"}
    )
    (run_dir / "result_summary.json").write_text(json.dumps(result, indent=2))
    update("Complete", 100)
    return result


def run_subprocess(command: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
