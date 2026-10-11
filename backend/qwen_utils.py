"""
Ollama (qwen2.5:7b) integration.

Qwen2.5 is used ONLY for two things:
  1. Extracting structured info (symptoms/duration/measurements/context) from
     the user's free-text description.
  2. Producing a plain-language EXPLANATION of the already-computed ML result.

It must never be asked to predict a disease, generate probabilities, or
invent values not present in the ML result / extracted text. Prompts below
enforce that explicitly.
"""
import json
import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_NAME = "qwen2.5:7b"

KNOWN_SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "shortness_of_breath",
    "chest_pain", "nausea", "dizziness", "joint_pain", "sore_throat",
    "runny_nose", "abdominal_pain", "frequent_urination", "excessive_thirst",
    "burning_urination", "wheezing", "heartburn", "palpitations",
]


def _call_ollama(prompt: str, timeout=60) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return json.dumps({"error": f"Ollama unavailable: {e}"})


def get_ollama_status() -> dict:
    """Return the detected local Ollama availability for the UI and backend health checks."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
        payload = resp.json() if hasattr(resp, "json") else {}
        models = payload.get("models", []) if isinstance(payload, dict) else []
        model_names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if name:
                    model_names.append(name)
        model_present = any(name == MODEL_NAME for name in model_names)
        return {
            "status": "online" if resp.ok else "offline",
            "available": resp.ok,
            "model_present": model_present,
            "model_name": MODEL_NAME,
        }
    except Exception:
        return {
            "status": "offline",
            "available": False,
            "model_present": False,
            "model_name": MODEL_NAME,
        }


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks some Qwen builds emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_from_text(description: str) -> dict:
    """Ask Qwen2.5 to extract structured facts from free text ONLY — no diagnosis."""
    prompt = f"""You are a text-extraction assistant. You must ONLY extract information
that is explicitly present in the patient's description below. Do NOT diagnose,
do NOT guess a disease, do NOT invent numbers that are not stated.

Respond with ONLY valid JSON (no other text) in this exact schema:
{{
  "symptoms_mentioned": {{"<symptom_name>": 0 or 1, ...}},
  "duration": "<free text duration or empty string>",
  "measurements_mentioned": {{"<name>": "<value as stated>", ...}},
  "other_context": "<short free text summary of anything else relevant>"
}}

Only use these exact symptom keys if mentioned (mark 1 if present, 0 if explicitly
denied/absent, omit if not mentioned at all): {", ".join(KNOWN_SYMPTOMS)}

Patient description:
\"\"\"{description}\"\"\"

JSON:"""
    raw = _strip_think(_call_ollama(prompt))
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else {"error": "Could not parse Qwen2.5 output", "raw": raw}
    except Exception:
        return {"error": "Could not parse Qwen2.5 output", "raw": raw}


def explain_result(patient_summary: dict, top_predictions: list, shap_summary: dict,
                    extracted_text_info: dict, contradictions: list) -> str:
    """Ask Qwen2.5 to explain an ALREADY-COMPUTED result in plain language."""
    prompt = f"""You are a medical-explanation assistant. The disease predictions below
were already computed by an XGBoost machine learning model — you must NOT change,
second-guess, or generate any new probabilities or diagnoses. Your only job is to
explain, in clear plain language for a patient, what the model found and why,
using the SHAP feature contributions provided. Mention any contradictions between
the patient's own description and the recorded data if given. End by reminding the
reader this is a statistical research tool, not a medical diagnosis, and to consult
a licensed clinician.

Computed top-3 predictions (from XGBoost, do not alter):
{json.dumps(top_predictions, indent=2)}

Top SHAP contributing features:
{json.dumps(shap_summary, indent=2)}

Information extracted from the patient's own description:
{json.dumps(extracted_text_info, indent=2)}

Detected contradictions between recorded data and description (if any):
{json.dumps(contradictions, indent=2)}

Write a concise (150-220 word) plain-language explanation now:"""
    return _strip_think(_call_ollama(prompt, timeout=90))


def cross_modal_consistency_check(csv_row: dict, extracted_info: dict) -> list:
    """Compare CSV symptom values against Qwen2.5-extracted text values."""
    contradictions = []
    text_symptoms = extracted_info.get("symptoms_mentioned", {}) if isinstance(extracted_info, dict) else {}
    for symptom, text_val in text_symptoms.items():
        if symptom not in csv_row:
            continue
        csv_val = csv_row.get(symptom)
        try:
            csv_val_int = int(csv_val)
        except (TypeError, ValueError):
            continue
        if csv_val_int != int(text_val):
            contradictions.append({
                "feature": symptom,
                "csv_value": csv_val_int,
                "text_value": int(text_val),
                "message": (
                    f"Recorded data shows '{symptom}' = {csv_val_int}, but the description "
                    f"indicates {'presence' if int(text_val) == 1 else 'absence'} of it."
                ),
            })
    return contradictions
