"""Pytest configuration for tests."""

import sys
from pathlib import Path

# Add parent directory (project root) to sys.path so that imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
