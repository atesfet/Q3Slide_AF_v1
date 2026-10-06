"""Create/update an isolated Conda environment, verify it, then run the app."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
PREFIX = RUNTIME / 'env'


@contextmanager
def setup_lock():
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / 'setup.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another launcher is setting up the environment. Wait for it to finish.') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def command(args):
    print(' '.join(str(a) for a in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', required=True)
    parser.add_argument('--setup-only', action='store_true')
    args, server_args = parser.parse_known_args()
    specification = ROOT / 'environment.yml'
    digest = hashlib.sha256(specification.read_bytes()).hexdigest()
    stamp = RUNTIME / 'environment.json'
    current = json.loads(stamp.read_text()) if stamp.exists() else {}
    conda = str(Path(args.conda).resolve())
    os.environ.setdefault('MPLCONFIGDIR', str(RUNTIME / 'cache' / 'matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('XDG_CACHE_HOME', str(RUNTIME / 'cache'))
    with setup_lock():
        if not (PREFIX / 'conda-meta' / 'history').exists():
            print('First launch: creating the dedicated Q3Slide environment. This can take several minutes.', flush=True)
            command([conda, 'env', 'create', '--prefix', str(PREFIX), '--file', str(specification), '--yes'])
        elif current.get('specification_sha256') != digest:
            print('Updating the application environment…', flush=True)
            command([conda, 'env', 'update', '--prefix', str(PREFIX), '--file', str(specification), '--prune'])
        verification = 'import numpy, scipy, tifffile, matplotlib, PIL; from oct_app.pipeline import inspect_geometry; print("Q3Slide environment verified.")'
        command([conda, 'run', '--no-capture-output', '--prefix', str(PREFIX), 'python', '-c', verification])
        stamp.write_text(json.dumps({'specification_sha256': digest, 'environment_prefix': str(PREFIX)}, indent=2))
    if not args.setup_only:
        return subprocess.call([conda, 'run', '--no-capture-output', '--prefix', str(PREFIX), 'python', '-u', '-m', 'oct_app.server', *server_args], cwd=ROOT)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'\nQ3Slide could not start: {exc}\nCheck your internet connection and write access to the application folder, then launch again.', file=sys.stderr)
        sys.exit(1)
