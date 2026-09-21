"""
Training pipeline: loads data/patients.csv, detects columns, preprocesses,
trains XGBoost, computes metrics, fits K-Means and Apriori rules, and saves
all artifacts to models/ for the FastAPI backend to load at runtime.

Usage:
    python train.py --csv ../data/patients.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml_utils import (
    LABEL_COL,
    MODEL_DIR,
    build_feature_frame,
    compute_metrics,
    detect_columns,
    run_apriori,
    run_kmeans,
    save_artifacts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "..", "data", "patients.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Training CSV must contain a '{LABEL_COL}' label column")

    columns = detect_columns(df)
    print("Detected columns:", json.dumps(columns, indent=2))

    X, imputers, encoders = build_feature_frame(df, columns, fit=True)
    feature_names = X.columns.tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # class weights to handle imbalance
    class_counts = np.bincount(y_train)
    total = len(y_train)
    sample_weight = np.array([total / (len(class_counts) * class_counts[label]) for label in y_train])

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print(f"Accuracy: {metrics['accuracy']:.3f}  F1(macro): {metrics['f1_macro']:.3f}  ROC-AUC: {metrics['roc_auc_macro']}")

    save_artifacts(model, label_encoder, imputers, encoders, columns, feature_names, metrics)

    # K-Means clustering fit on full feature set (for pattern analysis + convergence score)
    km, scaler, cluster_labels = run_kmeans(X, n_clusters=6)
    import joblib
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "kmeans_scaler.joblib"))

    cluster_summary = []
    df_with_cluster = df.copy()
    df_with_cluster["cluster"] = cluster_labels
    for c in sorted(set(cluster_labels)):
        sub = df_with_cluster[df_with_cluster["cluster"] == c]
        top_disease = sub[LABEL_COL].value_counts().idxmax()
        cluster_summary.append({
            "cluster": int(c),
            "size": int(len(sub)),
            "dominant_disease": top_disease,
            "avg_age": round(float(sub["age"].mean()), 1) if "age" in sub else None,
        })
    with open(os.path.join(MODEL_DIR, "cluster_summary.json"), "w") as f:
        json.dump(cluster_summary, f, indent=2)

    # Apriori association rules on binary symptom columns
    rules = run_apriori(df, columns["binary_columns"], min_support=0.04, min_confidence=0.35)
    with open(os.path.join(MODEL_DIR, "association_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Found {len(rules)} association rules")

    print("Training complete. Artifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
