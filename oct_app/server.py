from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .jobs import JobManager
from .pipeline import UserFacingError, inspect_geometry, suggest_fov
from .preview import PreviewManager


APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
PICKER_LOCK = threading.Lock()
PICKER_PROCESS = None


def stop_picker():
    process = PICKER_PROCESS
    if process is not None and process.poll() is None:
        process.terminate()


def choose_local_path(kind: str, initial_dir: str) -> str | None:
    global PICKER_PROCESS
    if kind not in {'image', 'folder'}:
        raise UserFacingError('Choose an image or folder.')
    if not PICKER_LOCK.acquire(blocking=False):
        raise UserFacingError('A file chooser is already open. Complete or cancel that dialog first.')
    try:
        process = subprocess.Popen([sys.executable, '-m', 'oct_app.file_picker', '--kind', kind,
                                 '--initial-dir', str(initial_dir)], cwd=APP_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        PICKER_PROCESS = process
        output, _ = process.communicate(timeout=900)
        if process.returncode != 0:
            raise UserFacingError('The system file chooser could not open. Enter the folder path manually instead.')
        data = json.loads(output)
        if data.get('error'):
            raise UserFacingError(data['error'])
        return data.get('path')
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise UserFacingError('The file chooser timed out or could not return a selection. Please try again.') from exc
    finally:
        stop_picker()
        PICKER_PROCESS = None
        PICKER_LOCK.release()


def scan_images(folder: Path) -> list[dict]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise UserFacingError(f"Input folder does not exist: {folder}")
    images = []
    for path in sorted(folder.rglob("*"), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            fov, reason = suggest_fov(path)
            images.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(folder)),
                    "size_bytes": path.stat().st_size,
                    "suggested_fov_um": fov,
                    "suggestion_reason": reason,
                }
            )
            if len(images) >= 1000:
                break
    return images


class OCTServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, input_dir: Path, output_dir: Path):
        super().__init__(address, handler)
        self.input_dir = input_dir.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.manager = JobManager(self.output_dir)
        self.previews = PreviewManager(APP_ROOT / '.runtime' / 'previews')
        self.started_at = time.time()


class Handler(BaseHTTPRequestHandler):
    server_version = "Q3SlideAF/1.0"

    @property
    def app(self) -> OCTServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400, detail: str | None = None) -> None:
        self._json({"ok": False, "error": message, "detail": detail}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise UserFacingError("Request is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            value = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise UserFacingError("The browser sent invalid JSON.") from exc
        if not isinstance(value, dict):
            raise UserFacingError("Expected a JSON object.")
        return value

    def _serve_path(self, path: Path, download_name: str | None = None) -> None:
        if not path.is_file():
            self._error("File not found", 404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        if download_name:
            quoted = urllib.parse.quote(download_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quoted}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if route == "/api/config":
            try:
                images = scan_images(self.app.input_dir)
                self._json(
                    {
                        "ok": True,
                        "version": "1.0.0",
                        "input_dir": str(self.app.input_dir),
                        "output_dir": str(self.app.output_dir),
                        "images": images,
                        "defaults": {
                            "fov_um": 700,
                            "auto_mode": True,
                            "strength": 1.0,
                            "top_k": 5,
                            "noise_sigma": 2.5,
                            "relative_depth_min_um": -10,
                            "relative_depth_max_um": 60,
                        },
                    }
                )
            except UserFacingError as exc:
                self._error(str(exc))
            return
        if route == "/api/jobs":
            self._json({"ok": True, "jobs": self.app.manager.list()})
            return
        if route.startswith('/api/previews/'):
            task_id = route.split('/')[3]
            task = self.app.previews.get(task_id)
            if not task:
                self._error('Unknown mosaic preview', 404)
            elif route.endswith('/image'):
                view = query.get('view', ['raw'])[0]
                if view not in {'raw', 'contrast'} or task['status'] != 'completed':
                    self._error('Preview image is not ready', 400)
                else:
                    self._serve_path(self.app.previews.root/task_id/f'{view}_projection.png')
            else:
                self._json({'ok': True, 'preview': task})
            return
        if route.startswith("/api/jobs/"):
            job_id = route.rsplit("/", 1)[-1]
            job = self.app.manager.get(job_id)
            if not job:
                self._error("Unknown analysis job", 404)
            else:
                self._json({"ok": True, "job": job.to_dict()})
            return
        if route == "/api/download":
            job_id = query.get("job", [""])[0]
            relative = query.get("file", [""])[0]
            job = self.app.manager.get(job_id)
            if not job:
                self._error("Unknown analysis job", 404)
                return
            try:
                target = (job.output_dir / relative).resolve()
                target.relative_to(job.output_dir.resolve())
            except (ValueError, OSError):
                self._error("Invalid output file path", 400)
                return
            inline = query.get("inline", ["0"])[0] == "1"
            self._serve_path(target, None if inline else target.name)
            return
        if route == "/api/health":
            self._json({"ok": True, "uptime_seconds": round(time.time() - self.app.started_at, 1)})
            return

        static_name = "index.html" if route == "/" else route.removeprefix("/")
        try:
            target = (STATIC_ROOT / static_name).resolve()
            target.relative_to(STATIC_ROOT.resolve())
        except (ValueError, OSError):
            self._error("Invalid path", 400)
            return
        self._serve_path(target)

    def do_POST(self) -> None:
        route = urllib.parse.urlparse(self.path).path
        try:
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if self.headers.get('Sec-Fetch-Site') == 'cross-site' or (origin and origin not in allowed):
                raise UserFacingError('Open Q3Slide from its local server address to use this API.')
            payload = self._read_json()
            if route in {'/api/inspect', '/api/previews', '/api/jobs'}:
                from .input_volume import prepare_input
                payload = prepare_input(payload)
            if route == '/api/browse':
                kind = str(payload.get('kind', 'image'))
                chosen = choose_local_path(kind, str(payload.get('initial_dir', self.app.input_dir)))
                if not chosen:
                    self._json({'ok': True, 'cancelled': True})
                    return
                path = Path(chosen).expanduser().resolve()
                if kind == 'image':
                    if not path.is_file() or path.suffix.lower() not in {'.tif', '.tiff'}:
                        raise UserFacingError('Please choose a TIFF image (.tif or .tiff).')
                    folder = path.parent
                    fov, reason = suggest_fov(path)
                    images = [{'name': path.name, 'path': str(path), 'relative_path': path.name,
                               'size_bytes': path.stat().st_size, 'suggested_fov_um': fov,
                               'suggestion_reason': reason}]
                else:
                    folder = path
                    images = scan_images(folder)
                self.app.input_dir = folder
                self._json({'ok': True, 'cancelled': False, 'input_dir': str(folder), 'images': images})
                return
            if route == "/api/scan":
                folder = Path(payload.get("folder", "")).expanduser().resolve()
                images = scan_images(folder)
                self.app.input_dir = folder
                self._json({"ok": True, "input_dir": str(folder), "images": images})
                return
            if route == "/api/inspect":
                image_path = Path(payload.get("image_path", ""))
                fov_um = float(payload.get("fov_um", 0))
                auto_mode = bool(payload.get("auto_mode", True))
                rows = None if auto_mode or not payload.get("rows") else int(payload["rows"])
                columns = None if auto_mode or not payload.get("columns") else int(payload["columns"])
                geometry = inspect_geometry(image_path, fov_um, rows, columns)
                self._json({"ok": True, "geometry": geometry.as_dict(), 'input_provenance': payload.get('input_provenance')})
                return
            if route == "/api/jobs":
                required = ["image_path", "fov_um", "action"]
                missing = [key for key in required if payload.get(key) in (None, "")]
                if missing:
                    raise UserFacingError(f"Missing required field: {', '.join(missing)}")
                if self.app.previews.busy():
                    raise UserFacingError('Wait for the reference mosaic preview to finish before starting analysis.')
                if payload.get('reference_patches') is not None:
                    provenance = self.app.previews.validate_selection(payload)
                    payload['reference_preview_provenance'] = provenance
                job = self.app.manager.create(payload)
                self._json({"ok": True, "job": job.to_dict()}, 202)
                return
            if route == '/api/previews':
                if any(job['status'] in {'running', 'queued', 'cancelling'} for job in self.app.manager.list()):
                    raise UserFacingError('Wait for the active analysis to finish before generating another preview.')
                self._json({'ok': True, 'preview': self.app.previews.create(payload)}, 202)
                return
            if route.startswith("/api/jobs/") and route.endswith("/cancel"):
                job_id = route.split("/")[3]
                if not self.app.manager.cancel(job_id):
                    self._error("Unknown analysis job", 404)
                else:
                    self._json({"ok": True})
                return
            if route == "/api/quit":
                self._json({"ok": True, "message": "OCT Coverslip Lab is shutting down."})

                def stop() -> None:
                    time.sleep(0.2)
                    self.app.manager.shutdown()
                    stop_picker()
                    self.app.shutdown()

                threading.Thread(target=stop, daemon=True).start()
                return
            if route == "/api/open-folder":
                job_id = str(payload.get("job_id", ""))
                job = self.app.manager.get(job_id)
                if not job:
                    self._error("Unknown analysis job", 404)
                    return
                if sys.platform == "darwin":
                    command = ["open", str(job.output_dir)]
                elif os.name == "nt":
                    command = ["explorer", str(job.output_dir)]
                else:
                    command = ["xdg-open", str(job.output_dir)]
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._json({"ok": True})
                return
            self._error("Unknown API route", 404)
        except UserFacingError as exc:
            self._error(str(exc), 400)
        except (TypeError, ValueError) as exc:
            self._error(f"Invalid parameter: {exc}", 400)
        except Exception as exc:
            self._error("The server could not complete this request.", 500, str(exc))


def choose_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {preferred} and {preferred + 19}")


def default_input_dir() -> Path:
    folder = APP_ROOT / 'input'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q3Slide AF locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "results")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    port = choose_port(args.host, args.port)
    server = OCTServer((args.host, port), Handler, args.input_dir, args.output_dir)
    url = f"http://{args.host}:{port}"
    print("\nQ3Slide AF v1.0")
    print(f"Open: {url}")
    print("Use the Quit Application button in the browser to stop the server.\n")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop_picker()
        server.manager.shutdown()
        server.server_close()
        print("OCT Coverslip Lab stopped.")


if __name__ == "__main__":
    main()
