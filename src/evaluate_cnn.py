"""Evaluate the trained CNN on held-out test actors."""

import json

import keras
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import EMOTIONS, PROCESSED_DIR, PROJECT_ROOT
from src.train_cnn import MODEL_FILE, NORM_FILE, prepare_cnn_input


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("CNN model not found. Run `python main.py train-cnn` first.")

    # Load features and split
    with np.load(PROCESSED_DIR / "features_dataset.npz", allow_pickle=False) as data:
        mfcc_all = data["mfcc"]
        labels = data["labels"].astype(np.int64)
        actor_ids = data["actor_ids"].astype(np.int64)
    with (PROCESSED_DIR / "actor_splits.json").open(encoding="utf-8") as fh:
        split = json.load(fh)

    test_mask = np.isin(actor_ids, split["test_actors"])
    X_all = prepare_cnn_input(mfcc_all)
    X_test = X_all[test_mask]
    y_test = labels[test_mask]

    # Normalize with training statistics
    norm = np.load(str(NORM_FILE))
    mean, std = norm["mean"], norm["std"]
    std = np.where(std < 1e-8, 1.0, std)
    X_test = (X_test - mean) / std

    # Load model and predict
    model = keras.models.load_model(str(MODEL_FILE))
    probabilities = model.predict(X_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    # Compute metrics
    results = {
        "model": "CNN with MFCC + delta + delta-delta (3-channel)",
        "test_actors": split["test_actors"],
        "test_samples": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test, predictions, target_names=EMOTIONS, output_dict=True, zero_division=0,
        ),
    }

    output = PROJECT_ROOT / "artifacts" / "cnn_test_results.json"
    with output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    cm = confusion_matrix(y_test, predictions)
    np.savetxt(
        PROJECT_ROOT / "artifacts" / "cnn_confusion_matrix.csv",
        cm, delimiter=",", fmt="%d",
    )

    print(json.dumps(
        {k: results[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "test_actors")},
        indent=2,
    ))


if __name__ == "__main__":
    main()

