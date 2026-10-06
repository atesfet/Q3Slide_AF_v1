from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from oct_app.pipeline import APP_ROOT, run_pipeline
from tests.test_core import write_synthetic_volume


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        image = temporary_path / "synthetic_100umFOV.tif"
        write_synthetic_volume(image)
        run_dir = temporary_path / "result"

        def update(stage: str, progress: int) -> None:
            print(f"{progress:3d}% {stage}")

        def run_command(command: list[str], label: str, progress: int) -> None:
            completed = subprocess.run(
                command,
                cwd=APP_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise RuntimeError(f"{label} failed\n{completed.stdout}")
            print(f"{progress:3d}% {label}")

        result = run_pipeline(
            {
                "image_path": str(image),
                "fov_um": 100,
                "action": "both",
                "auto_mode": True,
                "strength": 1.0,
                "interface_margin_um": 0,
                "top_k": 5,
                "noise_sigma": 1.5,
                "block_size_um": 40,
                "minimum_separation_um": 20,
                "reference_patches": [[0, 0], [0, 3], [3, 0], [3, 3]],
            },
            run_dir,
            run_command,
            update,
        )
        assert (run_dir / "synthetic_100umFOV_preprocessed.tif").is_file()
        assert (run_dir / "00_coverslip_spacing/coverslip_spacing_summary.json").is_file()
        assert (run_dir / "00_coverslip_spacing/patch_interface_report.csv").is_file()
        assert result["preprocessing"]["subtraction_strength"] == 1.0
        assert result["coverslip_spacing"]["valid_block_count"] > 0
        assert result['background_patches'] == [[0, 0], [0, 3], [3, 0], [3, 3]]
        assert result['background_selection_mode'] == 'reviewed-preview'
        assert (run_dir/'reference_selection.json').is_file()
        for item in result['files']:
            assert (run_dir/item['path']).is_file(), item['path']

        fallback_dir = temporary_path / "result_without_pair"

        def missing_pair_command(command: list[str], label: str, progress: int) -> None:
            if label == "coverslip-spacing":
                raise RuntimeError("coverslip-spacing exited with code 1\nCould not identify two distinct reflective interfaces")
            run_command(command, label, progress)

        fallback = run_pipeline(
            {
                "image_path": str(image),
                "fov_um": 100,
                "action": "both",
                "auto_mode": True,
                "strength": 1.0,
                "top_k": 5,
                "noise_sigma": 1.5,
            },
            fallback_dir,
            missing_pair_command,
            update,
        )
        assert (fallback_dir / "synthetic_100umFOV_preprocessed.tif").is_file()
        assert fallback["errors"][0]["code"] == "COVERSLIP_PAIR_NOT_DETECTED"
        assert fallback["preprocessing"]["projection_is_constrained_between_detected_coverslips"] is False
        assert (fallback_dir / "result_summary.json").is_file()

        try:
            run_pipeline(
                {"image_path": str(image), "fov_um": 100, "action": "spacing"},
                temporary_path / "spacing_only_without_pair",
                missing_pair_command,
                update,
            )
        except RuntimeError as exc:
            assert "Could not identify two distinct reflective interfaces" in str(exc)
        else:
            raise AssertionError("Spacing-only analysis must fail when the interface pair is missing")
        print("Smoke test passed, including missing-coverslip fallback")


if __name__ == "__main__":
    main()
