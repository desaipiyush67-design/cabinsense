from fastapi.testclient import TestClient

from cabinsense.api import app
from cabinsense.summary import summarize, trip_stats

EVENTS = [
    {"t": 600.0, "kind": "yawn", "value": 0.7},
    {"t": 1500.0, "kind": "microsleep", "value": 1.8},
    {"t": 900.0, "kind": "phone_use", "value": 1.2},
]


def test_trip_stats():
    s = trip_stats(EVENTS, 1800)
    assert s["duration_min"] == 30.0
    assert s["total_alerts"] == 3
    assert s["first_fatigue_sign_min"] == 10.0


def test_template_summary_without_llm(monkeypatch):
    monkeypatch.delenv("CABINSENSE_LLM_BASE_URL", raising=False)
    out = summarize(EVENTS, 1800)
    assert out["source"] == "template"
    assert "minute 10.0" in out["summary"] and "phone" in out["summary"]


def test_clean_trip():
    assert "no fatigue" in summarize([], 600)["summary"]


def test_api_summary(monkeypatch):
    monkeypatch.delenv("CABINSENSE_LLM_BASE_URL", raising=False)
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/trips/summary", json={"duration_s": 1800, "events": EVENTS})
    assert r.status_code == 200 and r.json()["stats"]["total_alerts"] == 3
