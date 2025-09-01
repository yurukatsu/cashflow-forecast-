import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="mlruns/artifacts/run", dir="~") as tmpdir:
    tmpdir_path = Path(tmpdir)
    tmpdir_path.mkdir(parents=True, exist_ok=True)
    print(f"Temporary directory created at: {tmpdir_path}")
