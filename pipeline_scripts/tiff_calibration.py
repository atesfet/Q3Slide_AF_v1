"""Read the axial calibration recorded by an ImageJ OCT TIFF export."""

from __future__ import annotations

import math


MICRON_UNITS = {"micron", "microns", "micrometer", "micrometers",
                "micrometre", "micrometres", "um", "µm", "μm"}


def read_z_spacing_um(imagej_metadata: dict | None) -> float:
    """Return calibrated slice spacing in µm; never infer a missing value."""
    metadata = imagej_metadata or {}
    raw = metadata.get("spacing")
    if raw is None:
        raise ValueError(
            "The TIFF is missing ImageJ Z spacing metadata. Re-export it with "
            "calibrated slice spacing in microns before measuring depth."
        )
    unit = str(metadata.get("unit", "")).strip().lower()
    if unit not in MICRON_UNITS:
        raise ValueError(
            f"The TIFF Z spacing unit is {unit or 'missing'!r}, not microns. "
            "Re-export it with micron calibration before measuring depth."
        )
    try:
        spacing = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The TIFF Z spacing {raw!r} is not a number.") from exc
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"The TIFF Z spacing {raw!r} must be finite and positive.")
    return spacing
