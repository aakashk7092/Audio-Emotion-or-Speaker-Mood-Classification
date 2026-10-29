
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import EMOTIONS, PROCESSED_DIR, PROJECT_ROOT


FEATURE_FILE = PROCESSED_DIR / "features_dataset.npz"
SPLIT_FILE = PROCESSED_DIR / "actor_splits.json"

MODEL_FILE = PROJECT_ROOT / "models" / "best_mfcc_cnn.keras"
NORMALIZATION_FILE = (
    PROJECT_ROOT / "artifacts" / "mfcc_normalization.npz"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def evaluate_model():
    # 1. Check required files
    for file_path in [
        FEATURE_FILE,
        SPLIT_FILE,
        MODEL_FILE,
        NORMALIZATION_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # 2. Load saved features and actor split
    with np.load(FEATURE_FILE, allow_pickle=False) as data:
        mfcc = data["mfcc"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        actor_ids = data["actor_ids"].astype(np.int64)

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        split_info = json.load(file)

    test_actors = split_info["test_actors"]

    test_indices = np.flatnonzero(
        np.isin(actor_ids, test_actors)
    )

    X_test = mfcc[test_indices][..., np.newaxis]
    y_test = labels[test_indices]

    # 3. Apply the exact training normalization
    with np.load(
        NORMALIZATION_FILE, allow_pickle=False
    ) as normalization:
        mean = float(normalization["mean"])
        std = float(normalization["std"])

    X_test = (X_test - mean) / std

    # 4. Load the saved best model
    model = tf.keras.models.load_model(str(MODEL_FILE))

    # 5. Predict test emotions
    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1,
    )
    predictions = np.argmax(probabilities, axis=1)

    # 6. Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=list(range(len(EMOTIONS))),
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(EMOTIONS))),
    )

    # 7. Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "test_samples": int(len(y_test)),
        "test_actors": test_actors,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "classification_report": report,
    }

    report_path = OUTPUT_DIR / "test_results.json"
    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(
        matrix,
        index=EMOTIONS,
        columns=EMOTIONS,
    ).to_csv(matrix_path, index_label="Actual")

    # 8. Display results
    print("\n========== TEST RESULTS ==========")
    print(f"Test samples:       {len(y_test)}")
    print(f"Test actors:        {test_actors}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro F1-score:     {macro_f1:.4f}")
    print(f"Balanced accuracy:  {balanced_accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    print(f"\nSaved results: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    evaluate_model()
