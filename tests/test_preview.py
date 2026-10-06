from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from oct_app.pipeline import UserFacingError, inspect_geometry, validate_reference_patches
from oct_app.preview import build_preview, PreviewManager, display_image
from tests.test_core import write_synthetic_volume


class PreviewTests(unittest.TestCase):
    def test_projection_includes_last_slice_and_tile_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume = np.ones((5, 100, 100), dtype=np.uint16)*10
            volume[-1] = np.arange(10000).reshape(100, 100)
            path = root/'scan.tif'
            tifffile.imwrite(path, volume, imagej=True,
                metadata={'axes': 'ZYX', 'spacing': 1.5, 'unit': 'micron'}, resolution=(.5, .5))
            result = build_preview({'image_path': str(path), 'fov_um': 100}, root/'preview')
            self.assertEqual(result['geometry']['rows'], 2)
            self.assertEqual(result['geometry']['columns'], 2)
            self.assertEqual(len(result['patches']), 4)
            actual = np.asarray(Image.open(root/'preview/raw_projection.png'))
            expected = np.asarray(display_image(volume.max(axis=0)))
            np.testing.assert_array_equal(actual, expected)

    def test_review_is_bound_to_image_and_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'scan.tif'
            write_synthetic_volume(path)
            config = {'image_path': str(path), 'fov_um': 100, 'reference_preview_id': 'abc',
                      'reference_patches': [[0, 0], [0, 3]], 'auto_mode': True}
            manager = PreviewManager(root/'previews')
            manager.tasks['abc'] = {'id': 'abc', 'status': 'completed', 'result': {
                'image_path': str(path.resolve()), 'file_mtime_ns': path.stat().st_mtime_ns,
                'geometry': inspect_geometry(path, 100).as_dict()}}
            manager.validate_selection(config)
            with self.assertRaises(UserFacingError):
                manager.validate_selection({**config, 'fov_um': 200})
            with self.assertRaisesRegex(UserFacingError, 'at least two'):
                manager.validate_selection({**config, 'reference_patches': [[0, 0]]})
            with self.assertRaisesRegex(UserFacingError, 'outside'):
                manager.validate_selection({**config, 'reference_patches': [[0, 0], [10, 10]]})
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
            with self.assertRaisesRegex(UserFacingError, 'changed'):
                manager.validate_selection(config)

    def test_reference_coordinates_are_distinct_and_in_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scan.tif'
            write_synthetic_volume(path)
            geometry = inspect_geometry(path, 100)
            self.assertEqual(validate_reference_patches([[0, 0], [0, 3], [0, 0]], geometry), [(0, 0), (0, 3)])
            with self.assertRaises(UserFacingError):
                validate_reference_patches([[True, 0], [0, 3]], geometry)
