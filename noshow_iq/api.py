"""
noshow_iq/api.py
NoShowIQ — FastAPI prediction service
Endpoints: GET /health | POST /predict | GET /history | GET /stats
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient, DESCENDING

# ── MongoDB setup ─────────────────────────────────────────────────────────────

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "noshow_iq"

client      = MongoClient(MONGO_URI)
db          = client[DB_NAME]
predictions = db["predictions"]
train_runs  = db["training_runs"]

# ── Model path ────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "noshow_model.joblib")

# ── Load model at startup ─────────────────────────────────────────────────────

model = None   # populated in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: model file not found at {MODEL_PATH}. "
              "Train first with main.py.")
    yield
    client.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NoShowIQ",
    description="Predict whether a patient will miss their clinic appointment.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response schemas ────────────────────────────────────────────────


class AppointmentRecord(BaseModel):
    """One raw appointment exactly as the clinic would send it."""
    Gender:           str   = Field(..., example="F")
    Age:              int   = Field(..., example=35)
    Neighbourhood:    str   = Field(..., example="JARDIM DA PENHA")
    Scholarship:      int   = Field(..., example=0)
    Hypertension:     int   = Field(..., example=0)
    Diabetes:         int   = Field(..., example=0)
    Alcoholism:       int   = Field(..., example=0)
    Handicap:         int   = Field(..., example=0)
    SMS_received:     int   = Field(..., example=1)
    ScheduledDay:     str   = Field(..., example="2016-04-29T08:00:00Z")
    AppointmentDay:   str   = Field(..., example="2016-04-29T00:00:00Z")


class PredictionResponse(BaseModel):
    risk_level:      str
    probability:     float
    recommendation:  str


# ── Helper: feature engineering (mirrors preprocess.py logic) ─────────────────

GENDER_MAP       = {"F": 0, "M": 1}
PART_OF_DAY_MAP  = {"Morning": 0, "Afternoon": 1, "Evening": 2}

# Neighbourhood encoding must match training — load from file if available,
# otherwise fall back to a hash-based ordinal (good-enough for inference).
NEIGHBOURHOOD_MAP_PATH = os.environ.get(
    "NEIGHBOURHOOD_MAP_PATH", "neighbourhood_map.joblib"
)
neighbourhood_map: dict = {}
if os.path.exists(NEIGHBOURHOOD_MAP_PATH):
    neighbourhood_map = joblib.load(NEIGHBOURHOOD_MAP_PATH)


def _part_of_day(hour: int) -> str:
    if 6 <= hour < 13:
        return "Morning"
    elif 13 <= hour < 19:
        return "Afternoon"
    return "Evening"


def _encode_neighbourhood(name: str) -> int:
    """Use saved map from training; fall back to stable hash ordinal."""
    if neighbourhood_map:
        return neighbourhood_map.get(name, -1)
    return abs(hash(name)) % 10_000


def engineer_features(record: AppointmentRecord) -> dict:
    """Turn a raw AppointmentRecord into the exact feature dict the model expects."""
    scheduled   = pd.to_datetime(record.ScheduledDay,  utc=True)
    appointment = pd.to_datetime(record.AppointmentDay, utc=True)

    days_in_advance  = (appointment.date() - scheduled.date()).days
    hours_in_advance = (appointment - scheduled).total_seconds() / 3600
    part_of_day_str  = _part_of_day(scheduled.hour)

    return {
        "Gender":           GENDER_MAP.get(record.Gender, 0),
        "Age":              record.Age,
        "Neighbourhood":    _encode_neighbourhood(record.Neighbourhood),
        "Scholarship":      record.Scholarship,
        "Hypertension":     record.Hypertension,
        "Diabetes":         record.Diabetes,
        "Alcoholism":       record.Alcoholism,
        "Handicap":         record.Handicap,
        "SMS_received":     record.SMS_received,
        "DaysInAdvance":    days_in_advance,
        "HoursInAdvance":   hours_in_advance,
        "PartOfDay":        PART_OF_DAY_MAP.get(part_of_day_str, 0),
    }


def _risk_label(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    elif prob >= 0.35:
        return "medium"
    return "low"


def _recommendation(risk: str) -> str:
    if risk == "high":
        return ("High no-show risk. Send an SMS reminder today and call the patient "
                "24 hours before the appointment.")
    elif risk == "medium":
        return ("Moderate risk. Schedule an automated SMS reminder 48 hours before "
                "the appointment.")
    return "Low risk. Standard reminder is sufficient."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness check — returns 200 when the service is up."""
    return {
        "status":    "ok",
        "model":     "loaded" if model else "not loaded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(record: AppointmentRecord):
    """
    Accept one appointment JSON, return:
    - risk_level      : 'low' | 'medium' | 'high'
    - probability     : float  (probability of no-show)
    - recommendation  : action string for clinic staff
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    # --- build features --
    try:
        features = engineer_features(record)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}")

    feature_df  = pd.DataFrame([features])
    try:
        prob_noshow = float(model.predict_proba(feature_df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    risk           = _risk_label(prob_noshow)
    recommendation = _recommendation(risk)

    # --- persist to MongoDB --
    doc = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "raw_input":        record.model_dump(),
        "cleaned_features": features,
        "risk_level":       risk,
        "probability":      round(prob_noshow, 4),
        "recommendation":   recommendation,
    }
    predictions.insert_one(doc)

    return PredictionResponse(
        risk_level=risk,
        probability=round(prob_noshow, 4),
        recommendation=recommendation,
    )


@app.get("/history", tags=["prediction"])
def history():
    """Return the last 20 predictions from MongoDB (newest first)."""
    cursor = (
        predictions
        .find({}, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(20)
    )
    return {"predictions": list(cursor)}


@app.get("/stats", tags=["analytics"])
def stats():
    """
    Aggregated stats computed entirely inside MongoDB (no Python arithmetic).
    Uses a single aggregation pipeline as required by the exam.
    """
    pipeline = [
        {
            "$facet": {
                # -- counts by risk level --
                "risk_counts": [
                    {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
                ],
                # -- total predictions & average probability --
                "overall": [
                    {
                        "$group": {
                            "_id":                None,
                            "total_predictions":  {"$sum": 1},
                            "average_probability":{"$avg": "$probability"},
                        }
                    }
                ],
            }
        }
    ]

    result = list(predictions.aggregate(pipeline))

    # -- last training run (separate collection, one lookup) --
    last_run = train_runs.find_one({}, {"_id": 0}, sort=[("timestamp", DESCENDING)])

    if not result:
        return {
            "total_predictions":   0,
            "high_risk_count":     0,
            "medium_risk_count":   0,
            "low_risk_count":      0,
            "average_probability": 0.0,
            "last_trained":        last_run.get("timestamp") if last_run else None,
        }

    facet        = result[0]
    risk_counts  = {item["_id"]: item["count"] for item in facet["risk_counts"]}
    overall      = facet["overall"][0] if facet["overall"] else {}

    return {
        "total_predictions":   overall.get("total_predictions", 0),
        "high_risk_count":     risk_counts.get("high",   0),
        "medium_risk_count":   risk_counts.get("medium", 0),
        "low_risk_count":      risk_counts.get("low",    0),
        "average_probability": round(overall.get("average_probability", 0.0), 4),
        "last_trained":        last_run.get("timestamp") if last_run else None,
    }
