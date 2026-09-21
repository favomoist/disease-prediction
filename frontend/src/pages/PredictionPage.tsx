import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadDataset, predict } from "../services/api";
import { useAppState } from "../state/AppState";

export default function PredictionPage() {
  const { dataset, setDataset, rowIndex, setRowIndex, description, setDescription, setResult } = useAppState();
  const [uploading, setUploading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const res = await uploadDataset(file);
      setDataset(res);
      setRowIndex(0);
      setResult(null);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handlePredict() {
    if (!dataset) return;
    setError(null);
    setPredicting(true);
    try {
      const res = await predict(dataset.dataset_id, rowIndex, description);
      setResult(res);
      navigate("/results");
    } catch (err: any) {
      setError(err.message || "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h2>1. Upload patient CSV</h2>
        <p className="small-muted">
          Upload a CSV with patient identification, demographics, binary symptoms and vitals/labs. See{" "}
          <code>data/patients_example.csv</code> for the expected format.
        </p>
        <input type="file" accept=".csv" onChange={handleFileChange} disabled={uploading} />
        {uploading && <p className="small-muted">Uploading and analyzing columns…</p>}
        {dataset && (
          <p className="small-muted" style={{ marginTop: 10 }}>
            Loaded <strong>{dataset.row_count}</strong> rows, {dataset.columns.length} columns. Detected{" "}
            {dataset.detected_columns.binary_columns.length} binary symptoms,{" "}
            {dataset.detected_columns.numerical_columns.length} numerical vitals/labs.
          </p>
        )}
      </div>

      {dataset && (
        <div className="card">
          <h2>2. Select patient row</h2>
          <label className="small-muted">Row index (0-based, max {dataset.row_count - 1})</label>
          <input
            type="number"
            min={0}
            max={dataset.row_count - 1}
            value={rowIndex}
            onChange={(e) => setRowIndex(Math.max(0, Math.min(dataset.row_count - 1, Number(e.target.value))))}
          />
        </div>
      )}

      {dataset && (
        <div className="card">
          <h2>3. Describe the situation (optional)</h2>
          <p className="small-muted">
            This free text is sent only to the local Qwen3 model to extract symptoms/duration/context and to explain the
            ML result. It never influences the ML prediction itself.
          </p>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. I've had a high fever and dry cough for the past 3 days, along with body aches..."
          />
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {dataset && (
        <button onClick={handlePredict} disabled={predicting}>
          {predicting ? "Running prediction…" : "Run Prediction"}
        </button>
      )}
    </div>
  );
}
