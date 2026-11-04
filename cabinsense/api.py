"""REST API for trip summaries.

    uvicorn cabinsense.api:app --reload
    POST /trips/summary  {"duration_s": 1800, "events": [{"t": 610.2, "kind": "yawn", "value": 0.71}]}
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .summary import summarize

app = FastAPI(title="CabinSense", version="0.1.0")


class EventIn(BaseModel):
    t: float
    kind: str
    value: float = 0.0
    detail: str = ""


class TripIn(BaseModel):
    duration_s: float = Field(gt=0)
    events: list[EventIn] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/trips/summary")
def trip_summary(trip: TripIn) -> dict:
    return summarize([e.model_dump() for e in trip.events], trip.duration_s)
