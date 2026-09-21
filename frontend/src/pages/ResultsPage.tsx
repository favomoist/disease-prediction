import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";
import { useAppState } from "../state/AppState";

export default function ResultsPage() {
  const { result } = useAppState();

  if (!result) {
    return (
      <div className="card">
        <h2>Results</h2>
        <p className="small-muted">Run a prediction from the Prediction page first.</p>
      </div>
    );
  }

  const shapChartData = [
    ...result.shap.positive_contributors.map((c) => ({ name: c.feature, value: c.shap_value })),
    ...result.shap.negative_contributors.map((c) => ({ name: c.feature, value: c.shap_value })),
  ].sort((a, b) => b.value - a.value);

  return (
    <div>
      <div className="card">
        <h2>Patient Information</h2>
        <div className="grid-2">
          {Object.entries(result.patient_info)
            .filter(([k]) => !k.toLowerCase().includes("name"))
            .map(([k, v]) => (
              <div key={k} className="small-muted">
                <strong>{k}:</strong> {String(v)}
              </div>
            ))}
        </div>
      </div>

      <div className="card">
        <h2>Top 3 Predicted Diseases (XGBoost probabilities)</h2>
        <table>
          <thead>
            <tr><th>Disease</th><th>Probability</th></tr>
          </thead>
          <tbody>
            {result.top_predictions.map((p) => (
              <tr key={p.disease}>
                <td>{p.disease}</td>
                <td>{(p.probability * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>SHAP Feature Contributions</h2>
        <p className="small-muted">Positive values push toward the predicted disease; negative values push away from it.</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={shapChartData} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={140} />
            <Tooltip />
            <Bar dataKey="value">
              {shapChartData.map((entry, i) => (
                <Cell key={i} fill={entry.value >= 0 ? "#16a34a" : "#dc2626"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="grid-2" style={{ marginTop: 12 }}>
          <div>
            <h3>Top positive contributors</h3>
            {result.shap.positive_contributors.map((c) => (
              <div key={c.feature} className="small-muted">
                <span className="badge positive">{c.feature}</span> value={c.value}, shap={c.shap_value.toFixed(3)}
              </div>
            ))}
          </div>
          <div>
            <h3>Top negative contributors</h3>
            {result.shap.negative_contributors.map((c) => (
              <div key={c.feature} className="small-muted">
                <span className="badge negative">{c.feature}</span> value={c.value}, shap={c.shap_value.toFixed(3)}
              </div>
            ))}
          </div>
        </div>
      </div>

      {Object.keys(result.extracted_text_info).length > 0 && (
        <div className="card">
          <h2>CSV vs. Description (Cross-Modal Consistency)</h2>
          <h3>Extracted from your description (Qwen3)</h3>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, background: "#f8f9fb", padding: 10, borderRadius: 6 }}>
            {JSON.stringify(result.extracted_text_info, null, 2)}
          </pre>
          <h3>Contradictions detected</h3>
          {result.contradictions.length === 0 ? (
            <p className="small-muted">No contradictions found.</p>
          ) : (
            <table>
              <thead><tr><th>Feature</th><th>CSV value</th><th>Text value</th><th>Message</th></tr></thead>
              <tbody>
                {result.contradictions.map((c) => (
                  <tr key={c.feature}>
                    <td>{c.feature}</td><td>{c.csv_value}</td><td>{c.text_value}</td><td>{c.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {result.qwen_explanation && (
        <div className="card">
          <h2>Qwen3 Plain-Language Explanation</h2>
          <p>{result.qwen_explanation}</p>
        </div>
      )}

      <div className="card">
        <h2>Cluster &amp; Association Rules</h2>
        {result.cluster && (
          <p className="small-muted">
            Assigned to cluster <strong>#{result.cluster.cluster}</strong> ({result.cluster.size} patients, dominant
            disease: {result.cluster.dominant_disease}), similarity score {result.cluster.similarity}.
          </p>
        )}
        <h3>Matched association rules ({result.matched_association_rules.length})</h3>
        {result.matched_association_rules.length === 0 ? (
          <p className="small-muted">No matching symptom association rules for this patient.</p>
        ) : (
          <table>
            <thead><tr><th>If symptoms</th><th>Then</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
            <tbody>
              {result.matched_association_rules.map((r, i) => (
                <tr key={i}>
                  <td>{r.antecedents.join(", ")}</td>
                  <td>{r.consequents.join(", ")}</td>
                  <td>{r.support}</td>
                  <td>{r.confidence}</td>
                  <td>{r.lift}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h2>Clinical Pattern Convergence Score (experimental, research-only)</h2>
        <div className="metrics-row">
          <div className="metric-tile">
            <span className="value">{result.convergence_score.score}</span>
            <span className="label">Score / 100</span>
          </div>
        </div>
        <table>
          <thead><tr><th>Component</th><th>Value (0-1)</th><th>Weight</th></tr></thead>
          <tbody>
            {Object.entries(result.convergence_score.components).map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{v}</td>
                <td>{result.convergence_score.weights[k]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small-muted">{result.convergence_score.disclaimer}</p>
      </div>

      <div className="disclaimer-banner">{result.disclaimer}</div>
    </div>
  );
}
