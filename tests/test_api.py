"""
tests/test_api.py
Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ── Sample input ──────────────────────────────────────────────────────────────

SAMPLE_RECORD = {
    "Gender":         "F",
    "Age":            35,
    "Neighbourhood":  "JARDIM DA PENHA",
    "Scholarship":    0,
    "Hypertension":   0,
    "Diabetes":       0,
    "Alcoholism":     0,
    "Handicap":       0,
    "SMS_received":   1,
    "ScheduledDay":   "2016-04-29T08:00:00Z",
    "AppointmentDay": "2016-05-04T00:00:00Z",
}


# ── Mock model factory ────────────────────────────────────────────────────────

def _make_mock_model(prob=0.72):
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - prob, prob]])
    return mock


# ── Mock MongoDB collections ──────────────────────────────────────────────────

def _make_mock_collections():
    fake_predictions = MagicMock()
    fake_predictions.insert_one.return_value = None
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = iter([])
    fake_predictions.find.return_value = mock_cursor
    fake_predictions.aggregate.return_value = iter([
        {
            "risk_counts": [
                {"_id": "high",   "count": 3},
                {"_id": "low",    "count": 7},
            ],
            "overall": [{"total_predictions": 10, "average_probability": 0.45}],
        }
    ])

    fake_train_runs = MagicMock()
    fake_train_runs.find_one.return_value = {"timestamp": "2026-04-01T09:00:00Z"}

    return fake_predictions, fake_train_runs


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    fake_predictions, fake_train_runs = _make_mock_collections()

    def fake_getitem(name):
        if name == "predictions":
            return fake_predictions
        return fake_train_runs

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(side_effect=fake_getitem)

    mock_mongo_client = MagicMock()
    mock_mongo_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_mongo_client.close = MagicMock()   # prevent real close

    with patch("noshow_iq.api.MongoClient", return_value=mock_mongo_client), \
         patch("noshow_iq.api.predictions", fake_predictions), \
         patch("noshow_iq.api.train_runs",  fake_train_runs), \
         patch("noshow_iq.api.model",       _make_mock_model()), \
         patch("os.path.exists",            return_value=False):

        from fastapi.testclient import TestClient
        from noshow_iq.api import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, fake_predictions, fake_train_runs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        c, *_ = client
        assert c.get("/health").status_code == 200

    def test_health_has_status_key(self, client):
        c, *_ = client
        assert c.get("/health").json().get("status") == "ok"


class TestPredict:
    def test_predict_returns_200(self, client):
        c, *_ = client
        with patch("noshow_iq.api.model", _make_mock_model()):
            resp = c.post("/predict", json=SAMPLE_RECORD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        c, *_ = client
        with patch("noshow_iq.api.model", _make_mock_model()):
            body = c.post("/predict", json=SAMPLE_RECORD).json()
        assert "risk_level"    in body
        assert "probability"   in body
        assert "recommendation" in body

    def test_predict_high_risk_label(self, client):
        c, *_ = client
        with patch("noshow_iq.api.model", _make_mock_model(prob=0.75)):
            body = c.post("/predict", json=SAMPLE_RECORD).json()
        assert body["risk_level"] == "high"

    def test_predict_low_risk_label(self, client):
        c, *_ = client
        with patch("noshow_iq.api.model", _make_mock_model(prob=0.10)):
            body = c.post("/predict", json=SAMPLE_RECORD).json()
        assert body["risk_level"] == "low"


class TestHistory:
    def test_history_returns_200(self, client):
        c, *_ = client
        assert c.get("/history").status_code == 200

    def test_history_has_predictions_key(self, client):
        c, *_ = client
        body = c.get("/history").json()
        assert "predictions" in body
        assert isinstance(body["predictions"], list)


class TestStats:
    def test_stats_returns_200(self, client):
        c, *_ = client
        assert c.get("/stats").status_code == 200

    def test_stats_has_required_keys(self, client):
        c, *_ = client
        body = c.get("/stats").json()
        assert "total_predictions"   in body
        assert "average_probability" in body
