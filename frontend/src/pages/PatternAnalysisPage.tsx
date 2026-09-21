import { useEffect, useState } from "react";
import { getPatterns, PatternsResponse } from "../services/api";

export default function PatternAnalysisPage() {
  const [data, setData] = useState<PatternsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPatterns()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <p className="small-muted">Loading pattern analysis…</p>;

  return (
    <div>
      <div className="card">
        <h2>K-Means Patient Clusters</h2>
        <table>
          <thead><tr><th>Cluster</th><th>Size</th><th>Dominant Disease</th><th>Avg Age</th></tr></thead>
          <tbody>
            {data.clusters.map((c) => (
              <tr key={c.cluster}>
                <td>#{c.cluster}</td>
                <td>{c.size}</td>
                <td>{c.dominant_disease}</td>
                <td>{c.avg_age ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Apriori Association Rules (symptom co-occurrence)</h2>
        <table>
          <thead><tr><th>If symptoms</th><th>Then</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
          <tbody>
            {data.association_rules.map((r, i) => (
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
        {data.association_rules.length === 0 && <p className="small-muted">No rules met the minimum support/confidence thresholds.</p>}
      </div>
    </div>
  );
}
