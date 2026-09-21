import { createContext, useContext, useState, ReactNode } from "react";
import type { UploadResponse, PredictionResult } from "../services/api";

interface AppState {
  dataset: UploadResponse | null;
  setDataset: (d: UploadResponse | null) => void;
  rowIndex: number;
  setRowIndex: (i: number) => void;
  description: string;
  setDescription: (s: string) => void;
  result: PredictionResult | null;
  setResult: (r: PredictionResult | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<UploadResponse | null>(null);
  const [rowIndex, setRowIndex] = useState(0);
  const [description, setDescription] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);

  return (
    <Ctx.Provider value={{ dataset, setDataset, rowIndex, setRowIndex, description, setDescription, result, setResult }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
