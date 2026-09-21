import { useEffect, useState } from "react";
import { getPreview, DetectedColumns } from "../services/api";
import { useAppState } from "../state/AppState";

export default function DatasetPage() {
  const { dataset } = useAppState();
  const [preview, setPreview] = useState<Record<string, unknown>[]>([]);
  const [columns, setColumns] = useState<DetectedColumns | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset) return;
    getPreview(dataset.dataset_id, 15)
      .then((res) => {
        setPreview(res.preview);
        setColumns(res.detected_columns);
      })
      .catch((err) => setError(err.message));
  }, [dataset]);

  if (!dataset) {
    return (
      <div className="card">
        <h2>Dataset</h2>
        <p className="small-muted">Upload a CSV on the Prediction page first.</p>
      </div>
    );
  }

  const cols = dataset.columns;

  return (
    <div>
      {error && <div className="error-box">{error}</div>}
      <div className="card">
        <h2>Dataset Overview</h2>
        <div className="metrics-row">
          <div className="metric-tile">
            <span className="value">{dataset.row_count}</span>
            <span className="label">Rows</span>
          </div>
          <div className="metric-tile">
            <span className="value">{cols.length}</span>
            <span className="label">Columns</span>
          </div>
          <div className="metric-tile">
            <span className="value">{dataset.has_label ? "Yes" : "No"}</span>
            <span className="label">Has 'disease' label</span>
          </div>
        </div>
      </div>

      {columns && (
        <div className="card">
          <h2>Detected Column Types</h2>
          <div className="grid-2">
            <div>
              <h3>Identification ({columns.id_columns.length})</h3>
              <div className="pill-list">{columns.id_columns.map((c) => <span className="pill" key={c}>{c}</span>)}</div>
            </div>
            <div>
              <h3>Binary Symptoms ({columns.binary_columns.length})</h3>
              <div className="pill-list">{columns.binary_columns.map((c) => <span className="pill" key={c}>{c}</span>)}</div>
            </div>
            <div>
              <h3>Categorical ({columns.categorical_columns.length})</h3>
              <div className="pill-list">{columns.categorical_columns.map((c) => <span className="pill" key={c}>{c}</span>)}</div>
            </div>
            <div>
              <h3>Numerical Vitals/Labs ({columns.numerical_columns.length})</h3>
              <div className="pill-list">{columns.numerical_columns.map((c) => <span className="pill" key={c}>{c}</span>)}</div>
            </div>
          </div>
        </div>
      )}

      <div className="card" style={{ overflowX: "auto" }}>
        <h2>Preview (first 15 rows)</h2>
        <table>
          <thead>
            <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i}>
                {cols.map((c) => <td key={c}>{String(row[c] ?? "")}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
