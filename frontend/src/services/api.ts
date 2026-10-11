const BASE_URL = "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export interface DetectedColumns {
  id_columns: string[];
  binary_columns: string[];
  categorical_columns: string[];
  numerical_columns: string[];
}

export interface UploadResponse {
  dataset_id: string;
  row_count: number;
  columns: string[];
  detected_columns: DetectedColumns;
  has_label: boolean;
  preview: Record<string, unknown>[];
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/api/dataset/upload`, { method: "POST", body: form });
  return handle(res);
}

export async function getPreview(datasetId: string, rows = 20) {
  const res = await fetch(`${BASE_URL}/api/dataset/${datasetId}/preview?rows=${rows}`);
  return handle<{ row_count: number; columns: string[]; detected_columns: DetectedColumns; preview: Record<string, unknown>[] }>(res);
}

export interface PredictionResult {
  disclaimer: string;
  patient_info: Record<string, unknown>;
  top_predictions: { disease: string; probability: number }[];
  shap: {
    positive_contributors: { feature: string; value: number; shap_value: number }[];
    negative_contributors: { feature: string; value: number; shap_value: number }[];
  };
  extracted_text_info: Record<string, unknown>;
  contradictions: { feature: string; csv_value: number; text_value: number; message: string }[];
  qwen_explanation: string;
  cluster: { cluster: number; size: number; dominant_disease: string; avg_age: number; similarity: number } | null;
  matched_association_rules: { antecedents: string[]; consequents: string[]; support: number; confidence: number; lift: number }[];
  convergence_score: {
    score: number;
    weights: Record<string, number>;
    components: Record<string, number>;
    disclaimer: string;
  };
}

export async function predict(datasetId: string, rowIndex: number, description: string): Promise<PredictionResult> {
  const res = await fetch(`${BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, row_index: rowIndex, description }),
  });
  return handle(res);
}

export interface PatternsResponse {
  clusters: { cluster: number; size: number; dominant_disease: string; avg_age: number | null }[];
  association_rules: { antecedents: string[]; consequents: string[]; support: number; confidence: number; lift: number }[];
}

export async function getPatterns(): Promise<PatternsResponse> {
  const res = await fetch(`${BASE_URL}/api/patterns`);
  return handle(res);
}

export interface MetricsResponse {
  metrics: {
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    roc_auc_macro: number | null;
    classification_report: Record<string, any>;
    confusion_matrix: number[][];
    class_names: string[];
  };
  columns: DetectedColumns;
  feature_names: string[];
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${BASE_URL}/api/model/metrics`);
  return handle(res);
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/api/health`);
  return handle<{
    status: string;
    model_trained: boolean;
    ollama_status: "online" | "offline";
    ollama_available: boolean;
    ollama_model_present: boolean;
    model_name: string;
    disclaimer: string;
  }>(res);
}
