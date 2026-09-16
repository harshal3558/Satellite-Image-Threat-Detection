"""
Unit tests for Satellite Image Threat Detection pipeline and web service.
"""

import sys
import numpy as np
import pytest

from src.SITP.exception import CustomException
from src.SITP.utils import normalize_to_uint8, seed_everything, tile_starts


def test_seed_everything():
    """Verify seed_everything sets reproducible state."""
    seed_everything(42)
    val1 = np.random.rand()
    seed_everything(42)
    val2 = np.random.rand()
    assert val1 == val2


def test_normalize_to_uint8():
    """Verify normalize_to_uint8 converts float arrays to uint8 in [0, 255]."""
    test_arr = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    norm = normalize_to_uint8(test_arr)
    assert norm.dtype == np.uint8
    assert norm.min() == 0
    assert norm.max() == 255


def test_tile_starts():
    """Verify tile_starts calculates sliding window intervals with boundary coverage."""
    starts = tile_starts(length=1000, tile_size=512, stride=256)
    assert starts[0] == 0
    assert starts[-1] == 1000 - 512
    assert len(starts) > 1

    # Length smaller than or equal to tile_size
    short_starts = tile_starts(length=300, tile_size=512, stride=256)
    assert short_starts == [0]


def test_custom_exception():
    """Verify CustomException captures error message and traceback details."""
    try:
        raise ValueError("Sample validation error for SITP test")
    except Exception as e:
        custom_exc = CustomException(e, sys)
        assert "Sample validation error for SITP test" in str(custom_exc)
        assert "line [" in str(custom_exc)


def test_flask_health_endpoint():
    """Verify Flask health check endpoint returns 200 and active status."""
    from application import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "satellite-threat-detection"


def test_flask_index_page():
    """Verify Flask landing page returns 200 and loads HTML."""
    from application import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Satellite Threat Intelligence" in response.data or b"SITP" in response.data
