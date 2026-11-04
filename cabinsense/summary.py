"""Post-trip safety summary from the event log.

If an OpenAI-compatible endpoint is configured (CABINSENSE_LLM_BASE_URL, CABINSENSE_LLM_API_KEY,
CABINSENSE_LLM_MODEL - works with OpenAI, Groq, Gemini's OpenAI endpoint or a local Ollama), an LLM
writes the summary from computed trip statistics. Otherwise a deterministic template is used, so the
pipeline never depends on network access.
"""
from __future__ import annotations

import json
import os
from collections import Counter

LABELS = {
    "drowsiness": "drowsiness (high PERCLOS)",
    "microsleep": "microsleep (long eye closure)",
    "yawn": "yawning",
    "distraction": "looking away from the road",
    "phone_use": "phone use",
    "no_face": "driver out of camera view",
}


def trip_stats(events: list[dict], duration_s: float) -> dict:
    counts = Counter(e["kind"] for e in events)
    first_fatigue = min((e["t"] for e in events if e["kind"] in ("drowsiness", "microsleep", "yawn")), default=None)
    return {
        "duration_min": round(duration_s / 60, 1),
        "events": dict(counts),
        "total_alerts": sum(counts.values()),
        "first_fatigue_sign_min": None if first_fatigue is None else round(first_fatigue / 60, 1),
    }


def template_summary(stats: dict) -> str:
    if not stats["total_alerts"]:
        return f"Trip of {stats['duration_min']} min: no fatigue or distraction alerts. Good drive."
    parts = [f"{n} x {LABELS.get(k, k)}" for k, n in sorted(stats["events"].items(), key=lambda kv: -kv[1])]
    advice = []
    if any(k in stats["events"] for k in ("drowsiness", "microsleep", "yawn")):
        when = stats["first_fatigue_sign_min"]
        advice.append(f"fatigue signs started around minute {when}; plan a break before that point on long drives")
    if "phone_use" in stats["events"]:
        advice.append("keep the phone mounted or out of reach while driving")
    if "distraction" in stats["events"]:
        advice.append("keep eyes on the road; use voice controls for navigation and media")
    if "no_face" in stats["events"]:
        advice.append("check the camera position so the driver's face stays in view")
    text = f"Trip of {stats['duration_min']} min with {stats['total_alerts']} alert(s): " + ", ".join(parts) + "."
    return text + (" " + "; ".join(advice).capitalize() + "." if advice else "")


def llm_summary(stats: dict) -> str | None:
    base, key, model = (os.getenv(k) for k in ("CABINSENSE_LLM_BASE_URL", "CABINSENSE_LLM_API_KEY", "CABINSENSE_LLM_MODEL"))
    if not (base and model):
        return None
    import httpx

    prompt = ("You are an in-car safety assistant. Write a short, friendly post-trip summary (max 4 sentences) "
              "for the driver from these statistics. Mention the most serious risk first and give one concrete tip. "
              f"Statistics (JSON): {json.dumps(stats)}")
    resp = httpx.post(f"{base.rstrip('/')}/chat/completions", timeout=30,
                      headers={"Authorization": f"Bearer {key}"} if key else {},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def summarize(events: list[dict], duration_s: float) -> dict:
    stats = trip_stats(events, duration_s)
    text, source = None, "template"
    try:
        text = llm_summary(stats)
        source = "llm" if text else source
    except Exception as exc:  # network/model errors fall back to the template
        source = f"template (llm error: {type(exc).__name__})"
    return {"stats": stats, "summary": text or template_summary(stats), "source": source}
