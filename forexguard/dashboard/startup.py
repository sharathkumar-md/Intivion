# Runs the pipeline on first launch if pre-computed data doesn't exist.
# Streamlit Community Cloud calls this before app.py via the import at top of app.py.

from pathlib import Path
import subprocess
import sys

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_FINAL_SCORES = _RAW_DIR / "final_scores.parquet"


def ensure_data():
    if _FINAL_SCORES.exists():
        return
    print("Pre-computed data not found — running pipeline (this takes ~5 min on first boot)...")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent.parent / "run_pipeline.py")],
        check=True,
    )
    print("Pipeline complete.")
