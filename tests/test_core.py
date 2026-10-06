from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from oct_app.pipeline import UserFacingError, inspect_geometry, parse_background_patches, suggest_fov
from pipeline_scripts.tiff_calibration import read_z_spacing_um


def write_synthetic_volume(path: Path, rows: int = 4, columns: int = 4, fov_um: float = 100.0) -> None:
    rng = np.random.default_rng(7)
    z_count = 48
    patch_px = 50
    height, width = rows * patch_px, columns * patch_px
    data = rng.normal(150, 22, (z_count, height, width)).astype(np.float32)
    z = np.arange(z_count, dtype=np.float32)[:, None, None]
    yy, xx = np.indices((height, width), dtype=np.float32)
    patch_row = (yy // patch_px).astype(int)
    patch_col = (xx // patch_px).astype(int)
    first_center = 10.0 + 0.55 * patch_row + 0.35 * patch_col
    second_center = first_center + 22.0
    illumination = 0.72 + 0.28 * np.exp(
        -((yy % patch_px - 24.5) ** 2 + (xx % patch_px - 24.5) ** 2) / (2 * 18.0**2)
    )
    data += 9500 * illumination[None] * np.exp(-0.5 * ((z - first_center[None]) / 1.05) ** 2)
    data += 7200 * illumination[None] * np.exp(-0.5 * ((z - second_center[None]) / 1.15) ** 2)
    tissue_xy = (yy > 52) & (yy < 148) & (xx > 52) & (xx < 148)
    texture = rng.uniform(0.5, 1.5, (height, width)).astype(np.float32)
    for tissue_z in range(16, 23):
        data[tissue_z] += tissue_xy * texture * (2100 + 500 * np.sin(xx / 5.0))
    data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        metadata={"axes": "ZYX", "unit": "micron", "spacing": 2.0},
        resolution=(0.5, 0.5),
    )


class CoreTests(unittest.TestCase):
    def test_fov_suggestion(self) -> None:
        self.assertEqual(suggest_fov(Path("sample_700umFOV.tif"))[0], 700.0)
        self.assertEqual(suggest_fov(Path("sample_1mmFOV.tif"))[0], 1000.0)
        self.assertEqual(suggest_fov(Path("sample_10um_H&E.tif"))[0], 700.0)

    def test_background_parser(self) -> None:
        self.assertEqual(parse_background_patches("0,5; 0,6 1,9"), [(0, 5), (0, 6), (1, 9)])

    def test_geometry_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_100umFOV.tif"
            write_synthetic_volume(path)
            geometry = inspect_geometry(path, 100.0)
            self.assertEqual((geometry.rows, geometry.columns), (4, 4))
            self.assertEqual(geometry.patch_shape_yx, (50, 50))
            self.assertEqual(geometry.voxel_size_um_zyx, (2.0, 2.0, 2.0))

    def test_axial_spacing_is_read_not_assumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibrated_100umFOV.tif"
            tifffile.imwrite(
                path, np.zeros((3, 50, 50), dtype=np.uint16), imagej=True,
                metadata={"axes": "ZYX", "unit": "micron", "spacing": 1.4338522576276844},
                resolution=(0.5, 0.5),
            )
            geometry = inspect_geometry(path, 100.0)
            self.assertAlmostEqual(geometry.voxel_size_um_zyx[0], 1.4338522576276844)

    def test_missing_axial_spacing_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "uncalibrated_100umFOV.tif"
            tifffile.imwrite(
                path, np.zeros((3, 50, 50), dtype=np.uint16), imagej=True,
                metadata={"axes": "ZYX", "unit": "micron"}, resolution=(0.5, 0.5),
            )
            with self.assertRaisesRegex(UserFacingError, "missing ImageJ Z spacing"):
                inspect_geometry(path, 100.0)

    def test_invalid_axial_calibration_fails_closed(self) -> None:
        for metadata in (
            {"unit": "micron", "spacing": 0},
            {"unit": "micron", "spacing": float("nan")},
            {"unit": "pixel", "spacing": 2},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                read_z_spacing_um(metadata)


if __name__ == "__main__":
    unittest.main()
