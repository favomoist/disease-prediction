"""
Shared ML utilities: column detection, preprocessing, training helpers,
SHAP explanation, clustering, association rules and the convergence score.
Kept intentionally compact and reused by both train.py and main.py.
"""
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ID_NAME_HINTS = ("id", "name", "patient_id", "patient_name")
LABEL_COL = "disease"


def detect_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into id, binary, categorical, numerical (excludes label)."""
    id_cols, binary_cols, categorical_cols, numerical_cols = [], [], [], []
    for col in df.columns:
        if col == LABEL_COL:
            continue
        lname = col.lower()
        if any(h in lname for h in ID_NAME_HINTS):
            id_cols.append(col)
            continue
        series = df[col]
        non_null = series.dropna()
        is_text_dtype = (
            series.dtype == object
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if is_text_dtype:
            uniq = set(non_null.unique().tolist())
            if uniq.issubset({0, 1, "0", "1"}):
                binary_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            uniq_vals = non_null.unique()
            if set(np.unique(uniq_vals.astype(float))).issubset({0.0, 1.0}) and len(uniq_vals) <= 2:
                binary_cols.append(col)
            else:
                numerical_cols.append(col)
    return {
        "id_columns": id_cols,
        "binary_columns": binary_cols,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
    }


def build_feature_frame(df: pd.DataFrame, columns: dict[str, list[str]],
                         imputers: dict[str, Any] | None = None,
                         encoders: dict[str, Any] | None = None,
                         fit: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    """Impute + encode features. Returns (feature_df, imputers, encoders)."""
    imputers = imputers or {}
    encoders = encoders or {}
    feat_cols = columns["binary_columns"] + columns["categorical_columns"] + columns["numerical_columns"]
    work = df[feat_cols].copy()

    # numerical: median impute
    num_cols = columns["numerical_columns"]
    if num_cols:
        if fit:
            imp = SimpleImputer(strategy="median")
            work[num_cols] = imp.fit_transform(work[num_cols])
            imputers["numerical"] = imp
        else:
            imp = imputers["numerical"]
            work[num_cols] = imp.transform(work[num_cols])

    # binary: mode impute, ensure numeric 0/1
    bin_cols = columns["binary_columns"]
    for c in bin_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if bin_cols:
        if fit:
            imp = SimpleImputer(strategy="most_frequent")
            work[bin_cols] = imp.fit_transform(work[bin_cols])
            imputers["binary"] = imp
        else:
            imp = imputers["binary"]
            work[bin_cols] = imp.transform(work[bin_cols])

    # categorical: mode impute + label encode
    cat_cols = columns["categorical_columns"]
    if cat_cols:
        if fit:
            imp = SimpleImputer(strategy="most_frequent")
            work[cat_cols] = imp.fit_transform(work[cat_cols])
            imputers["categorical"] = imp
        else:
            imp = imputers["categorical"]
            work[cat_cols] = imp.transform(work[cat_cols])
        for c in cat_cols:
            if fit:
                le = LabelEncoder()
                work[c] = le.fit_transform(work[c].astype(str))
                encoders[c] = le
            else:
                le = encoders[c]
                work[c] = work[c].astype(str).map(
                    lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
                )

    return work[feat_cols], imputers, encoders


def save_artifacts(model, label_encoder, imputers, encoders, columns, feature_names, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(imputers, os.path.join(MODEL_DIR, "imputers.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "columns.json"), "w") as f:
        json.dump(columns, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    imputers = joblib.load(os.path.join(MODEL_DIR, "imputers.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "columns.json")) as f:
        columns = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, label_encoder, imputers, encoders, columns, feature_names, metrics


def compute_metrics(y_true, y_pred, y_proba, label_encoder) -> dict:
    class_names = label_encoder.classes_.tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def shap_explain(model, feature_row: pd.DataFrame, feature_names: list[str], predicted_class_idx: int, top_n=6):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row)
    # shap_values shape for multiclass xgboost sklearn API: (n_classes, n_samples, n_features) or list
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class_idx][0]
    elif shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class_idx]
    else:
        class_shap = shap_values[0]

    contributions = [
        {"feature": feature_names[i], "value": float(feature_row.iloc[0, i]), "shap_value": float(class_shap[i])}
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda c: c["shap_value"], reverse=True)
    positive = [c for c in contributions if c["shap_value"] > 0][:top_n]
    negative = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda c: c["shap_value"])[:top_n]
    return {"positive_contributors": positive, "negative_contributors": negative}


def run_kmeans(feature_df: pd.DataFrame, n_clusters=5):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return km, scaler, labels


def assign_cluster(km, scaler, feature_row: pd.DataFrame):
    """Assign the row to its nearest K-Means cluster and compute a 0-1
    similarity score relative to how spread out the cluster centers are
    (a simple, transparent normalization — not a statistical guarantee)."""
    scaled = scaler.transform(feature_row)
    cluster = int(km.predict(scaled)[0])
    center = km.cluster_centers_[cluster]
    dist = float(np.linalg.norm(scaled[0] - center))
    # reference scale: largest pairwise distance between any two cluster centers
    centers = km.cluster_centers_
    if len(centers) > 1:
        pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        scale = float(np.max(pairwise)) or 1.0
    else:
        scale = 1.0
    similarity = float(max(0.0, 1 - dist / (scale + 1e-6)))
    return cluster, similarity


def run_apriori(df: pd.DataFrame, binary_cols: list[str], min_support=0.05, min_confidence=0.4):
    transactions = df[binary_cols].fillna(0).astype(int)
    te_df = transactions.astype(bool)
    freq = apriori(te_df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return []
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("confidence", ascending=False).head(15)
    result = []
    for _, r in rules.iterrows():
        result.append({
            "antecedents": sorted(list(r["antecedents"])),
            "consequents": sorted(list(r["consequents"])),
            "support": round(float(r["support"]), 3),
            "confidence": round(float(r["confidence"]), 3),
            "lift": round(float(r["lift"]), 3),
        })
    return result


def match_association_rules(rules: list[dict], patient_symptoms: dict[str, int]) -> list[dict]:
    present = {s for s, v in patient_symptoms.items() if v == 1}
    matches = []
    for r in rules:
        if set(r["antecedents"]).issubset(present):
            matches.append(r)
    return matches


def convergence_score(xgb_confidence: float, shap_alignment: float, rule_evidence: float, cluster_similarity: float):
    """
    Transparent, explainable, research-only score combining four signals
    (each 0-1) with fixed, disclosed weights. NOT a validated clinical score.
    """
    weights = {"xgb_confidence": 0.4, "shap_alignment": 0.25, "rule_evidence": 0.2, "cluster_similarity": 0.15}
    components = {
        "xgb_confidence": round(xgb_confidence, 3),
        "shap_alignment": round(shap_alignment, 3),
        "rule_evidence": round(rule_evidence, 3),
        "cluster_similarity": round(cluster_similarity, 3),
    }
    score = sum(components[k] * weights[k] for k in weights)
    return {
        "score": round(score * 100, 1),
        "weights": weights,
        "components": components,
        "disclaimer": "Experimental research-only score. Not a medically validated diagnostic metric.",
    }
