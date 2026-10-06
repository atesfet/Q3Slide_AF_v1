from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .pipeline import APP_ROOT, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OCT Coverslip Lab pipeline without the website."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--fov-um", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("preprocess", "spacing", "both"), default="both")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--block-size-um", type=float, default=100.0)
    parser.add_argument("--minimum-separation-um", type=float, default=20.0)
    parser.add_argument("--refractive-index", type=float)
    args = parser.parse_args()

    config = {
        "image_path": str(args.image.resolve()),
        "fov_um": args.fov_um,
        "action": args.action,
        "auto_mode": True,
        "strength": args.strength,
        "block_size_um": args.block_size_um,
        "minimum_separation_um": args.minimum_separation_um,
        "refractive_index": args.refractive_index,
    }

    def update(stage: str, progress: int) -> None:
        print(f"[{progress:3d}%] {stage}", flush=True)

    def run_command(command: list[str], label: str, progress: int) -> None:
        print(f"[{label}] starting", flush=True)
        completed = subprocess.run(command, cwd=APP_ROOT)
        if completed.returncode:
            raise RuntimeError(f"{label} exited with code {completed.returncode}")
        print(f"[{progress:3d}%] {label} complete", flush=True)

    result = run_pipeline(config, args.output_dir.resolve(), run_command, update)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "result": result}, indent=2))


if __name__ == "__main__":
    main()
