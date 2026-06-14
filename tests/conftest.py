"""Shared test fixtures for BaseStation-MAS."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable from tests/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture
def sample_kpi_array():
    """Generate a dummy (18, 128) KPI array for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.randn(18, 128).astype(np.float32)
