import { useEffect, useState } from "react";
import { HashRouter, Routes, Route, NavLink } from "react-router-dom";
import { AppStateProvider } from "./state/AppState";
import PredictionPage from "./pages/PredictionPage";
import DatasetPage from "./pages/DatasetPage";
import ResultsPage from "./pages/ResultsPage";
import PatternAnalysisPage from "./pages/PatternAnalysisPage";
import ModelPerformancePage from "./pages/ModelPerformancePage";
import { getHealth } from "./services/api";

const links = [
  { to: "/", label: "Prediction" },
  { to: "/dataset", label: "Dataset" },
  { to: "/results", label: "Results" },
  { to: "/patterns", label: "Pattern Analysis" },
  { to: "/performance", label: "Model Performance" },
];

export default function App() {
  const [ollamaStatus, setOllamaStatus] = useState<"online" | "offline">("offline");

  useEffect(() => {
    let active = true;

    getHealth()
      .then((health) => {
        if (!active) return;
        setOllamaStatus(health.ollama_status === "online" ? "online" : "offline");
      })
      .catch(() => {
        if (active) setOllamaStatus("offline");
      });

    return () => {
      active = false;
    };
  }, []);

  const isOnline = ollamaStatus === "online";

  return (
    <AppStateProvider>
      <HashRouter>
        <div className="app-shell">
          <aside className="sidebar">
            <div className="sidebar-header">
              <h1>AI Disease Prediction</h1>
              <div className={`agent-status ${isOnline ? "online" : "offline"}`} aria-label={isOnline ? "Ollama connected" : "Ollama offline"}>
                <span className="agent-dot" />
                <span>Ollama</span>
                <span className="agent-state">{isOnline ? "Connected" : "Offline"}</span>
              </div>
            </div>
            <nav>
              {links.map((l) => (
                <NavLink key={l.to} to={l.to} end={l.to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
                  {l.label}
                </NavLink>
              ))}
            </nav>
          </aside>
          <main className="main-content">
            <div className="disclaimer-banner">
              Research &amp; education tool only. Outputs are statistical ML predictions, not medical diagnoses. Always
              consult a licensed clinician.
            </div>
            <Routes>
              <Route path="/" element={<PredictionPage />} />
              <Route path="/dataset" element={<DatasetPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/patterns" element={<PatternAnalysisPage />} />
              <Route path="/performance" element={<ModelPerformancePage />} />
            </Routes>
          </main>
        </div>
      </HashRouter>
    </AppStateProvider>
  );
}
