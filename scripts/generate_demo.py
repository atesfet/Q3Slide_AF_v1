"""Generate a small synthetic volume for testing the workbench, never patient data."""
from pathlib import Path
from tests.test_core import write_synthetic_volume

if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'input' / 'demo_100umFOV.tif'
    destination.parent.mkdir(exist_ok=True)
    write_synthetic_volume(destination)
    print(f'Synthetic test volume created: {destination}')
