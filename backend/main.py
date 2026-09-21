"""
FastAPI backend for the local-only AI Disease Prediction demo.
Compact, single-file router with shared helpers in ml_utils.py / qwen_utils.py.
"""
import io
import json
import os
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_utils import (
    LABEL_COL,
    MODEL_DIR,
    assign_cluster,
    build_feature_frame,
    detect_columns,
    load_artifacts,
    match_association_rules,
    shap_explain,
)
from qwen_utils import cross_modal_consistency_check, explain_result, extract_from_text

app = FastAPI(title="AI Disease Prediction API (local/research only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for uploaded datasets, keyed by dataset_id (single-user local app)
DATASETS: dict[str, pd.DataFrame] = {}

DISCLAIMER = (
    "These results are statistical machine-learning outputs generated for research "
    "and educational purposes. They are NOT a medical diagnosis or treatment "
    "recommendation. Always consult a licensed healthcare professional."
)


def _artifacts_ready():
    return os.path.exists(os.path.join(MODEL_DIR, "xgb_model.json"))


def _require_artifacts():
    if not _artifacts_ready():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run backend/train.py first.")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained": _artifacts_ready(), "disclaimer": DISCLAIMER}


# ---------------------------------------------------------------------------
# Dataset upload / preview
# ---------------------------------------------------------------------------
@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    columns = detect_columns(df)
    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "columns": list(df.columns),
        "detected_columns": columns,
        "has_label": LABEL_COL in df.columns,
        "preview": json.loads(df.head(10).to_json(orient="records")),
    }


@app.get("/api/dataset/{dataset_id}/preview")
def dataset_preview(dataset_id: str, rows: int = 20):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")
    df = DATASETS[dataset_id]
    columns = detect_columns(df)
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "detected_columns": columns,
        "preview": json.loads(df.head(rows).to_json(orient="records")),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    dataset_id: str
    row_index: int = 0
    description: str = ""


@app.post("/api/predict")
def predict(req: PredictRequest):
    _require_artifacts()
    if req.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")
    df = DATASETS[req.dataset_id]
    if req.row_index < 0 or req.row_index >= len(df):
        raise HTTPException(status_code=400, detail="row_index out of range")

    model, label_encoder, imputers, encoders, columns, feature_names, metrics = load_artifacts()

    row_df = df.iloc[[req.row_index]].copy()
    # Ensure required feature columns exist; fill missing ones with NaN
    for c in columns["binary_columns"] + columns["categorical_columns"] + columns["numerical_columns"]:
        if c not in row_df.columns:
            row_df[c] = np.nan

    X_row, _, _ = build_feature_frame(row_df, columns, imputers=imputers, encoders=encoders, fit=False)
    X_row = X_row[feature_names]

    proba = model.predict_proba(X_row)[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top_predictions = [
        {"disease": label_encoder.classes_[i], "probability": round(float(proba[i]), 4)}
        for i in top_idx
    ]
    predicted_class_idx = int(top_idx[0])

    shap_summary = shap_explain(model, X_row, feature_names, predicted_class_idx)

    # Qwen3 text extraction + consistency check (only if description provided)
    extracted_info, contradictions, explanation = {}, [], ""
    if req.description.strip():
        extracted_info = extract_from_text(req.description)
        csv_row_dict = row_df.iloc[0].to_dict()
        contradictions = cross_modal_consistency_check(csv_row_dict, extracted_info)
        explanation = explain_result(
            patient_summary=csv_row_dict,
            top_predictions=top_predictions,
            shap_summary=shap_summary,
            extracted_text_info=extracted_info,
            contradictions=contradictions,
        )

    # Pattern analysis: cluster + association rules relevant to this patient
    cluster_info, rule_matches = None, []
    try:
        km = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "kmeans_scaler.joblib"))
        cluster, similarity = assign_cluster(km, scaler, X_row)
        with open(os.path.join(MODEL_DIR, "cluster_summary.json")) as f:
            cluster_summary = json.load(f)
        cluster_info = next((c for c in cluster_summary if c["cluster"] == cluster), None)
        cluster_info = {**(cluster_info or {}), "similarity": round(similarity, 3)}

        with open(os.path.join(MODEL_DIR, "association_rules.json")) as f:
            rules = json.load(f)
        symptom_vals = {c: int(row_df.iloc[0][c]) for c in columns["binary_columns"] if c in row_df.columns and not pd.isna(row_df.iloc[0][c])}
        rule_matches = match_association_rules(rules, symptom_vals)
    except FileNotFoundError:
        pass

    # Convergence score components
    xgb_confidence = float(proba[predicted_class_idx])
    pos_sum = sum(c["shap_value"] for c in shap_summary["positive_contributors"])
    neg_sum = sum(abs(c["shap_value"]) for c in shap_summary["negative_contributors"])
    shap_alignment = float(pos_sum / (pos_sum + neg_sum)) if (pos_sum + neg_sum) > 0 else 0.5
    rule_evidence = min(1.0, len(rule_matches) / 3) if rule_matches else 0.0
    cluster_similarity = cluster_info["similarity"] if cluster_info else 0.5

    from ml_utils import convergence_score
    conv = convergence_score(xgb_confidence, shap_alignment, rule_evidence, cluster_similarity)

    patient_info = {k: v for k, v in row_df.iloc[0].to_dict().items()}

    return {
        "disclaimer": DISCLAIMER,
        "patient_info": patient_info,
        "top_predictions": top_predictions,
        "shap": shap_summary,
        "extracted_text_info": extracted_info,
        "contradictions": contradictions,
        "qwen_explanation": explanation,
        "cluster": cluster_info,
        "matched_association_rules": rule_matches,
        "convergence_score": conv,
    }


# ---------------------------------------------------------------------------
# Pattern analysis (global, not per-patient)
# ---------------------------------------------------------------------------
@app.get("/api/patterns")
def patterns():
    _require_artifacts()
    with open(os.path.join(MODEL_DIR, "cluster_summary.json")) as f:
        clusters = json.load(f)
    with open(os.path.join(MODEL_DIR, "association_rules.json")) as f:
        rules = json.load(f)
    return {"clusters": clusters, "association_rules": rules}


# ---------------------------------------------------------------------------
# Model performance / metadata
# ---------------------------------------------------------------------------
@app.get("/api/model/metrics")
def model_metrics():
    _require_artifacts()
    _, _, _, _, columns, feature_names, metrics = load_artifacts()
    return {"metrics": metrics, "columns": columns, "feature_names": feature_names}


# ---------------------------------------------------------------------------
# Standalone Qwen3 endpoints (used internally by /api/predict, exposed too)
# ---------------------------------------------------------------------------
class DescriptionRequest(BaseModel):
    description: str


@app.post("/api/qwen/extract")
def qwen_extract(req: DescriptionRequest):
    return extract_from_text(req.description)


class ConsistencyRequest(BaseModel):
    dataset_id: str
    row_index: int
    description: str


@app.post("/api/consistency-check")
def consistency_check(req: ConsistencyRequest):
    if req.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")
    df = DATASETS[req.dataset_id]
    if req.row_index < 0 or req.row_index >= len(df):
        raise HTTPException(status_code=400, detail="row_index out of range")
    extracted = extract_from_text(req.description)
    contradictions = cross_modal_consistency_check(df.iloc[req.row_index].to_dict(), extracted)
    return {"extracted_text_info": extracted, "contradictions": contradictions}
