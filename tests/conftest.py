"""Pytest configuration.

This project is often run directly from a source checkout (especially in CI).
Ensure the repository root is on `sys.path` so `import dimreduce4gpu` works
whether or not the package has been installed in editable mode.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
