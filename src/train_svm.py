"""Train a validation-selected SVM baseline for speaker-independent SER."""

import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROCESSED_DIR, PROJECT_ROOT, RANDOM_SEED


FEATURE_FILE = PROCESSED_DIR / "features_dataset.npz"
SPLIT_FILE = PROCESSED_DIR / "actor_splits.json"
MODEL_FILE = PROJECT_ROOT / "models" / "best_ser_svm.joblib"
SUMMARY_FILE = PROJECT_ROOT / "artifacts" / "training_summary_svm.json"


def summarize(feature_maps: np.ndarray) -> np.ndarray:
    """Convert time-frequency maps into robust fixed-size SER descriptors.

    Enhanced version: includes delta and delta-delta statistics for richer
    temporal information than the original mean/std-only descriptor.
    """
    # Static statistics across time (9 per band)
    stats = [
        feature_maps.mean(axis=2),
        feature_maps.std(axis=2),
        feature_maps.min(axis=2),
        feature_maps.max(axis=2),
        np.percentile(feature_maps, 10, axis=2),
        np.percentile(feature_maps, 25, axis=2),
        np.percentile(feature_maps, 50, axis=2),
        np.percentile(feature_maps, 75, axis=2),
        np.percentile(feature_maps, 90, axis=2),
    ]

    # First-order delta — how rapidly the acoustic profile changes (5 per band)
    delta = np.diff(feature_maps, axis=2)
    stats.extend([
        delta.mean(axis=2),
        delta.std(axis=2),
        np.percentile(delta, 10, axis=2),
        np.percentile(delta, 50, axis=2),
        np.percentile(delta, 90, axis=2),
    ])

    # Second-order delta — acceleration of change (3 per band)
    delta2 = np.diff(delta, axis=2)
    stats.extend([
        delta2.mean(axis=2),
        delta2.std(axis=2),
        delta2.max(axis=2) - delta2.min(axis=2),  # range
    ])

    return np.concatenate(stats, axis=1).astype(np.float32)


def main():
    if not FEATURE_FILE.exists() or not SPLIT_FILE.exists():
        raise FileNotFoundError("Run `python -m src.features` and `python -m src.split_data` first.")

    with np.load(FEATURE_FILE, allow_pickle=False) as data:
        # MFCC captures compact vocal-tract cues; log-Mel retains spectral energy.
        features = np.concatenate([summarize(data["mfcc"]), summarize(data["log_mel"])], axis=1)
        labels = data["labels"].astype(np.int64)
        actor_ids = data["actor_ids"].astype(np.int64)
    with SPLIT_FILE.open(encoding="utf-8") as file:
        split = json.load(file)

    train = np.isin(actor_ids, split["train_actors"])
    validation = np.isin(actor_ids, split["validation_actors"])
    test = np.isin(actor_ids, split["test_actors"])
    if not np.all(train | validation | test):
        raise ValueError("Actor split does not cover every sample.")

    print(f"Feature dimensions: {features.shape[1]}", flush=True)
    print(f"Training: {train.sum()} | Validation: {validation.sum()} | Test: {test.sum()}", flush=True)

    # Extended hyperparameter grid for better model selection
    candidates = [
        (c, gamma)
        for c in (0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)
        for gamma in ("scale", "auto", 0.0001, 0.0005, 0.001, 0.005)
    ]

    best = None
    for c_value, gamma in candidates:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED)),
        ])
        model.fit(features[train], labels[train])
        predictions = model.predict(features[validation])
        score = f1_score(labels[validation], predictions, average="macro", zero_division=0)
        print(f"C={c_value:<5} gamma={str(gamma):<6} val_macro_f1={score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, c_value, gamma)

    _, c_value, gamma = best
    print(f"\nBest hyperparameters: C={c_value}, gamma={gamma}, val_f1={best[0]:.4f}", flush=True)

    # Refit only with training + validation actors after hyperparameters are fixed;
    # test actors remain untouched until final evaluation.
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED)),
    ])
    final_model.fit(features[train | validation], labels[train | validation])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    SUMMARY_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)

    validation_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED)),
    ])
    validation_model.fit(features[train], labels[train])
    validation_predictions = validation_model.predict(features[validation])
    summary = {
        "model": "Enhanced RBF SVM with MFCC and log-Mel statistical descriptors",
        "best_params": {"C": c_value, "gamma": gamma},
        "feature_dimensions": int(features.shape[1]),
        "validation_accuracy": float(accuracy_score(labels[validation], validation_predictions)),
        "validation_macro_f1": float(f1_score(labels[validation], validation_predictions, average="macro", zero_division=0)),
        "train_samples": int(train.sum()),
        "validation_samples": int(validation.sum()),
        "test_samples_reserved": int(test.sum()),
        "test_used_for_model_selection": False,
        "model_path": str(MODEL_FILE),
    }
    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
