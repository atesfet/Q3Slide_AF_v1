from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from oct_app.input_volume import prepare_input
from oct_app.pipeline import UserFacingError, inspect_geometry
from oct_app.preview import build_preview
from tests.test_core import write_synthetic_volume


class InputVolumeTests(unittest.TestCase):
    def test_sidecar_infers_layout_and_calibration_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'reconstruction.tiff'
            values = np.arange(6*4*8, dtype=np.uint16).reshape(6,4,8)
            tifffile.imwrite(raw, values, metadata=None, photometric='minisblack', compression='deflate')
            metadata = {a: {'values': (np.arange(n)*step).tolist(), 'units': 'mm'}
                        for a,n,step in [('y',6,.002), ('z',4,.00143), ('x',8,.002)]}
            sidecar = Path(str(raw)+'.json')
            sidecar.write_text(json.dumps({'metadata': metadata}))
            with patch('oct_app.pipeline.APP_ROOT', root):
                prepared = prepare_input({'image_path': str(raw)})
                np.testing.assert_array_equal(tifffile.imread(prepared['image_path']), values.swapaxes(0,1))
                provenance = prepared['input_provenance']
                self.assertEqual(provenance['interpreted_source_layout'], 'YZX')
                self.assertEqual(provenance['calibration_source'], 'reconstruction sidecar')
                np.testing.assert_allclose(provenance['voxel_size_um_zyx'], [1.43,2,2])
                metadata['z']['values'][2] += .01
                sidecar.write_text(json.dumps({'metadata': metadata}))
                with self.assertRaisesRegex(UserFacingError, 'uniformly increasing'):
                    prepare_input({'image_path': str(raw)})

    def test_generic_axes_require_explicit_interpretation_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'generic.tiff'
            values = np.arange(6*4*8, dtype=np.uint16).reshape(6,4,8)
            tifffile.imwrite(raw, values, metadata=None, photometric='minisblack')
            with tifffile.TiffFile(raw) as tif:
                self.assertEqual(tif.series[0].axes, 'IYX')
            with self.assertRaisesRegex(UserFacingError, 'explicitly choose'):
                prepare_input({'image_path': str(raw)})
            with self.assertRaisesRegex(UserFacingError, 'physical calibration'):
                prepare_input({'image_path': str(raw), 'input_layout': 'ZYX'})
            for voxels in ([2, '', 3], [2, 0, 3], [2, float('nan'), 3]):
                with self.subTest(voxels=voxels), self.assertRaises(UserFacingError):
                    prepare_input({'image_path': str(raw), 'input_layout': 'ZYX', 'voxel_size_um_zyx': voxels})
            with patch('oct_app.pipeline.APP_ROOT', root):
                for layout in ['ZYX', 'YZX']:
                    config = {'image_path': str(raw), 'input_layout': layout, 'voxel_size_um_zyx': [1.43, 2, 2]}
                    prepared = prepare_input(config)
                    actual = tifffile.memmap(prepared['image_path'])
                    expected = values if layout == 'ZYX' else values.swapaxes(0,1)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, values.dtype)
                    self.assertEqual(prepare_input(config)['image_path'], prepared['image_path'])
                    self.assertEqual(prepared['input_provenance']['source_axes'], 'IYX')
                    del actual
            np.testing.assert_array_equal(tifffile.imread(raw), values)

    def test_resliced_raw_stack_flows_into_same_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrated = root/'calibrated.tif'
            raw = root/'resliced_100umFOV.tiff'
            write_synthetic_volume(calibrated)
            original = tifffile.imread(calibrated)
            tifffile.imwrite(raw, original.swapaxes(0,1), metadata=None, photometric='minisblack')
            with patch('oct_app.pipeline.APP_ROOT', root):
                prepared = prepare_input({'image_path': str(raw), 'input_layout': 'YZX',
                    'voxel_size_um_zyx': [2,2,2], 'fov_um': 100})
                geometry = inspect_geometry(Path(prepared['image_path']), 100)
                self.assertEqual(geometry.shape_zyx, original.shape)
                self.assertEqual((geometry.rows, geometry.columns), (4,4))
                result = build_preview(prepared, root/'preview')
                self.assertEqual(result['geometry']['shape_zyx'], list(original.shape))
                self.assertEqual(len(result['patches']), 16)
