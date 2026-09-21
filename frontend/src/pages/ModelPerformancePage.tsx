import { useEffect, useState } from "react";
import { getMetrics, MetricsResponse } from "../services/api";

export default function ModelPerformancePage() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <p className="small-muted">Loading model metrics…</p>;

  const m = data.metrics;
  const classNames = m.class_names;

  return (
    <div>
      <div className="card">
        <h2>Overall Metrics (held-out test set)</h2>
        <div className="metrics-row">
          <div className="metric-tile"><span className="value">{(m.accuracy * 100).toFixed(1)}%</span><span className="label">Accuracy</span></div>
          <div className="metric-tile"><span className="value">{(m.precision_macro * 100).toFixed(1)}%</span><span className="label">Precision (macro)</span></div>
          <div className="metric-tile"><span className="value">{(m.recall_macro * 100).toFixed(1)}%</span><span className="label">Recall (macro)</span></div>
          <div className="metric-tile"><span className="value">{(m.f1_macro * 100).toFixed(1)}%</span><span className="label">F1 (macro)</span></div>
          <div className="metric-tile">
            <span className="value">{m.roc_auc_macro !== null ? (m.roc_auc_macro * 100).toFixed(1) + "%" : "N/A"}</span>
            <span className="label">ROC-AUC (macro, OvR)</span>
          </div>
        </div>
      </div>

      <div className="card" style={{ overflowX: "auto" }}>
        <h2>Per-Class Report</h2>
        <table>
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>
            {classNames.map((c) => {
              const row = m.classification_report[c];
              if (!row) return null;
              return (
                <tr key={c}>
                  <td>{c}</td>
                  <td>{row.precision.toFixed(2)}</td>
                  <td>{row.recall.toFixed(2)}</td>
                  <td>{row["f1-score"].toFixed(2)}</td>
                  <td>{row.support}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ overflowX: "auto" }}>
        <h2>Confusion Matrix</h2>
        <table>
          <thead>
            <tr>
              <th>Actual \ Predicted</th>
              {classNames.map((c) => <th key={c}>{c.slice(0, 6)}</th>)}
            </tr>
          </thead>
          <tbody>
            {m.confusion_matrix.map((row, i) => (
              <tr key={i}>
                <td><strong>{classNames[i].slice(0, 10)}</strong></td>
                {row.map((val, j) => (
                  <td key={j} style={{ background: i === j ? "#dcfce7" : undefined }}>{val}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Feature Summary</h2>
        <p className="small-muted">{data.feature_names.length} features used for training (identification columns excluded).</p>
        <div className="pill-list">{data.feature_names.map((f) => <span className="pill" key={f}>{f}</span>)}</div>
      </div>
    </div>
  );
}
