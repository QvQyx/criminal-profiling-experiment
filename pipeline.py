"""Reproducible learning experiment on explicitly synthetic records."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES = ["Age", "Income", "Crime Count"]
TARGET = "Crime Type"

def rule_labels(counts):
    return np.where(np.asarray(counts) > 3, "Violent Crime",
                    np.where(np.asarray(counts) > 1, "Theft", "No or Minor Crime"))

def generate_data(n=500, seed=42):
    # Preserve the original generator's sequence for comparison.
    rng = np.random.RandomState(seed)
    data = pd.DataFrame({"Age": rng.randint(18, 60, n),
                         "Income": rng.normal(5000, 1500, n),
                         "Crime Count": rng.poisson(1.5, n)})
    data[TARGET] = rule_labels(data["Crime Count"])
    return data

def fit_model(data):
    """Shared training path for the app and offline evaluation."""
    if data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not np.array_equal(data[TARGET], rule_labels(data["Crime Count"])):
        raise ValueError("Dataset does not follow the documented synthetic rule")
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42,
        stratify=data[TARGET])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # String labels avoid a separate encoder.
    return model, X_test, y_test

def train():
    data = pd.read_csv(ROOT / "simulated_crime_data.csv")
    model, X_test, y_test = fit_model(data)
    predicted = model.predict(X_test)
    baseline = rule_labels(X_test["Crime Count"])
    report = {
        "data_source": "synthetic; labels assigned by a known count rule",
        "seed": 42, "train_rows": len(data) - len(X_test), "test_rows": len(X_test),
        "class_counts": data[TARGET].value_counts().to_dict(),
        "accuracy": accuracy_score(y_test, predicted),
        "rule_baseline_accuracy": accuracy_score(y_test, baseline),
        "majority_baseline_accuracy": float((y_test == data.loc[~data.index.isin(X_test.index), TARGET].mode()[0]).mean()),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "class_order": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=model.classes_).tolist(),
        "feature_importance": dict(zip(FEATURES, model.feature_importances_.tolist())),
        "sklearn_version": sklearn.__version__,
        "interpretation": "Measures recovery of an artificial rule, not real-world crime prediction."
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "sklearn_version": sklearn.__version__},
                ROOT / "models/model.joblib")
    (ROOT / "reports/metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "train"])
    args = parser.parse_args()
    if args.command == "generate":
        generate_data().to_csv(ROOT / "simulated_crime_data.csv", index=False)
    else:
        train()
