"""
Unit tests for Satellite Image Threat Detection pipeline and web service.
"""

import numpy as np
import pytest

from src.SITP.utils import normalize_to_uint8, seed_everything


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
