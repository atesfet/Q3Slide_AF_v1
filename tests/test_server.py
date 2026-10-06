from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from oct_app.server import OCTServer, Handler
from tests.test_core import write_synthetic_volume


class ServerTests(unittest.TestCase):
    def test_preview_review_api_and_changed_grid_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'scan_100umFOV.tif'
            write_synthetic_volume(path)
            server = OCTServer(('127.0.0.1', 0), Handler, root, root/'results')
            server.previews.root = root/'previews'
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def request(route, body=None):
                req = Request(base+route, data=json.dumps(body).encode() if body is not None else None,
                              headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=10) as response:
                    return json.load(response)
            try:
                self.assertTrue(request('/api/health')['ok'])
                self.assertEqual(request('/api/config')['images'][0]['suggested_fov_um'], 100)
                config = {'image_path': str(path), 'fov_um': 100, 'auto_mode': True}
                preview = request('/api/previews', config)['preview']
                for _ in range(100):
                    preview = request('/api/previews/'+preview['id'])['preview']
                    if preview['status'] != 'running':
                        break
                    time.sleep(.05)
                self.assertEqual(preview['status'], 'completed', preview.get('error'))
                with urlopen(base+f"/api/previews/{preview['id']}/image?view=raw", timeout=10) as response:
                    self.assertEqual(response.headers['Content-Type'], 'image/png')
                    self.assertTrue(response.read().startswith(b'\x89PNG'))
                server.manager.create = Mock(return_value=Mock(to_dict=lambda: {'id': 'test-job'}))
                edited = [[0, 0], [3, 3]]
                payload = {**config, 'action': 'both', 'reference_preview_id': preview['id'], 'reference_patches': edited}
                self.assertTrue(request('/api/jobs', payload)['ok'])
                captured = server.manager.create.call_args.args[0]
                self.assertEqual(captured['reference_patches'], edited)
                self.assertIn('reference_preview_provenance', captured)
                with self.assertRaises(HTTPError) as changed:
                    request('/api/jobs', {**payload, 'fov_um': 200})
                self.assertEqual(changed.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
