"""
Synthetic patient dataset generator for the AI Disease Prediction demo.
Generates a CSV with identification fields, demographics, binary symptoms,
numerical vitals/labs, some missing values, and realistic disease->symptom
correlations across 11 disease classes.

Usage:
    python generate_dataset.py --rows 5000 --out patients.csv
"""
import argparse
import random
import numpy as np
import pandas as pd

DISEASES = [
    "Influenza", "Type2Diabetes", "Hypertension", "CommonCold",
    "Asthma", "CoronaryArteryDisease", "GERD", "UrinaryTractInfection",
    "Migraine", "Pneumonia", "Healthy",
]

SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "shortness_of_breath",
    "chest_pain", "nausea", "dizziness", "joint_pain", "sore_throat",
    "runny_nose", "abdominal_pain", "frequent_urination", "excessive_thirst",
    "burning_urination", "wheezing", "heartburn", "palpitations",
]

# Probability that a symptom is present (1) given a disease.
DISEASE_SYMPTOM_PROB = {
    "Influenza": {"fever": .9, "cough": .8, "fatigue": .85, "headache": .6, "joint_pain": .5, "sore_throat": .4},
    "Type2Diabetes": {"fatigue": .7, "frequent_urination": .85, "excessive_thirst": .8, "dizziness": .3},
    "Hypertension": {"headache": .5, "dizziness": .45, "palpitations": .3, "chest_pain": .2},
    "CommonCold": {"cough": .7, "runny_nose": .85, "sore_throat": .7, "fatigue": .4, "fever": .3},
    "Asthma": {"shortness_of_breath": .85, "wheezing": .8, "cough": .6, "chest_pain": .3},
    "CoronaryArteryDisease": {"chest_pain": .8, "shortness_of_breath": .6, "fatigue": .5, "palpitations": .4},
    "GERD": {"heartburn": .85, "chest_pain": .3, "nausea": .4, "abdominal_pain": .4},
    "UrinaryTractInfection": {"burning_urination": .85, "frequent_urination": .75, "abdominal_pain": .4, "fever": .3},
    "Migraine": {"headache": .9, "nausea": .5, "dizziness": .4},
    "Pneumonia": {"fever": .8, "cough": .85, "shortness_of_breath": .7, "chest_pain": .4, "fatigue": .6},
    "Healthy": {},
}

# baseline vitals ranges (normal) and shifts per disease (mean shift, std)
BASE_VITALS = {
    "systolic_bp": (118, 10), "diastolic_bp": (76, 8), "heart_rate": (75, 10),
    "temperature_c": (36.8, 0.3), "respiratory_rate": (16, 2), "spo2": (98, 1),
    "glucose_mg_dl": (95, 12), "cholesterol_mg_dl": (180, 25), "bmi": (24, 3),
}

VITAL_SHIFTS = {
    "Influenza": {"temperature_c": (1.5, 0.4), "heart_rate": (10, 5)},
    "Type2Diabetes": {"glucose_mg_dl": (90, 30), "bmi": (4, 2)},
    "Hypertension": {"systolic_bp": (25, 10), "diastolic_bp": (15, 6)},
    "Asthma": {"respiratory_rate": (5, 2), "spo2": (-3, 1.5)},
    "CoronaryArteryDisease": {"cholesterol_mg_dl": (40, 20), "heart_rate": (8, 5)},
    "Pneumonia": {"temperature_c": (1.8, 0.5), "spo2": (-4, 2), "respiratory_rate": (6, 2)},
    "UrinaryTractInfection": {"temperature_c": (0.7, 0.3)},
}

FIRST_NAMES = ["Aarav", "Vivaan", "Ishaan", "Ananya", "Diya", "Kabir", "Meera",
               "Rohan", "Sanya", "Aditya", "Priya", "Karan", "Neha", "Arjun",
               "Sara", "Vikram", "Anika", "Rahul", "Pooja", "Dev"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Nair", "Gupta", "Reddy", "Khan",
              "Patel", "Singh", "Rao", "Das", "Mehta", "Joshi", "Kapoor"]


def gen_patient(pid: int, rng: random.Random, np_rng: np.random.Generator):
    disease = rng.choices(DISEASES, weights=[1.1, 1, 1, 1.3, 0.9, 0.7, 0.9, 0.9, 0.9, 0.6, 1.8])[0]
    age = int(np.clip(np_rng.normal(45, 18), 1, 95))
    gender = rng.choice(["Male", "Female"])
    height_cm = np.clip(np_rng.normal(165 if gender == "Female" else 175, 8), 140, 200)
    weight_kg = np.clip(np_rng.normal(65 if gender == "Female" else 75, 14), 35, 150)

    row = {
        "patient_id": f"P{pid:05d}",
        "patient_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "age": age,
        "gender": gender,
        "height_cm": round(height_cm, 1),
        "weight_kg": round(weight_kg, 1),
    }

    # symptoms
    probs = DISEASE_SYMPTOM_PROB.get(disease, {})
    for s in SYMPTOMS:
        p = probs.get(s, 0.04)  # small baseline noise chance for any symptom
        row[s] = 1 if rng.random() < p else 0

    # vitals with disease-specific shifts
    shifts = VITAL_SHIFTS.get(disease, {})
    for vital, (mean, std) in BASE_VITALS.items():
        shift_mean, shift_std = shifts.get(vital, (0, 0))
        value = np_rng.normal(mean + shift_mean, std + shift_std * 0.3)
        if vital == "bmi":
            value = weight_kg / ((height_cm / 100) ** 2)
            value += shift_mean * 0.3
        row[vital] = round(float(value), 1)

    row["disease"] = disease
    return row


def inject_missing(df: pd.DataFrame, frac=0.03):
    rng = np.random.default_rng(42)
    numeric_cols = [c for c in BASE_VITALS.keys()]
    for col in numeric_cols:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5200)
    parser.add_argument("--out", type=str, default="patients.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    rows = [gen_patient(i + 1, rng, np_rng) for i in range(args.rows)]
    df = pd.DataFrame(rows)
    df = inject_missing(df, frac=0.03)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df["disease"].value_counts())

    # small example file for upload-testing (20 rows, no disease/label column,
    # matching what a real user would upload for prediction)
    example = df.drop(columns=["disease"]).sample(20, random_state=1)
    example.to_csv("patients_example.csv", index=False)
    print("Wrote patients_example.csv (20 rows, unlabeled)")


if __name__ == "__main__":
    main()
