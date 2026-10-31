"""Evaluate the validation-selected SVM only once on held-out actors."""

import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import EMOTIONS, PROCESSED_DIR, PROJECT_ROOT
from src.train_svm import MODEL_FILE, summarize


def main():
    with np.load(PROCESSED_DIR / "features_dataset.npz", allow_pickle=False) as data:
        features = np.concatenate([summarize(data["mfcc"]), summarize(data["log_mel"])], axis=1)
        labels, actor_ids = data["labels"], data["actor_ids"]
    with (PROCESSED_DIR / "actor_splits.json").open(encoding="utf-8") as file:
        split = json.load(file)
    test = np.isin(actor_ids, split["test_actors"])
    model = joblib.load(MODEL_FILE)
    predictions = model.predict(features[test])
    results = {
        "model": "RBF SVM with MFCC and log-Mel statistical descriptors",
        "test_actors": split["test_actors"],
        "test_samples": int(test.sum()),
        "accuracy": float(accuracy_score(labels[test], predictions)),
        "macro_f1": float(f1_score(labels[test], predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test], predictions)),
        "classification_report": classification_report(labels[test], predictions, target_names=EMOTIONS, output_dict=True, zero_division=0),
    }
    output = PROJECT_ROOT / "artifacts" / "svm_test_results.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    np.savetxt(PROJECT_ROOT / "artifacts" / "svm_confusion_matrix.csv", confusion_matrix(labels[test], predictions), delimiter=",", fmt="%d")
    print(json.dumps({key: results[key] for key in ("accuracy", "macro_f1", "balanced_accuracy", "test_actors")}, indent=2))


if __name__ == "__main__":
    main()
