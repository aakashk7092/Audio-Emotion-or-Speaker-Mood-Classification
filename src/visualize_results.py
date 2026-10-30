
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EMOTIONS, PROJECT_ROOT

CSV_PATH = PROJECT_ROOT / "artifacts" / "confusion_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "plots"


def plot_confusion_matrix():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, index_col=0)

    matrix = df.to_numpy(dtype=np.int64)

    if matrix.shape != (len(EMOTIONS), len(EMOTIONS)):
        raise ValueError(
            f"Expected an 8x8 matrix, got {matrix.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        linewidths=0.5,
    )

    plt.title("Baseline MFCC CNN - Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "baseline_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved to: {output_path}")


def plot_training_history():
    import pandas as pd
    import matplotlib.pyplot as plt

    history_path = PROJECT_ROOT / "artifacts" / "training_history.csv"

    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    history = pd.read_csv(history_path)

    required_columns = [
        "epoch", "accuracy", "val_accuracy", "loss", "val_loss"
    ]

    missing = [
        col for col in required_columns if col not in history.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy graph
    axes[0].plot(
        history["epoch"] + 1,
        history["accuracy"],
        label="Training Accuracy"
    )
    axes[0].plot(
        history["epoch"] + 1,
        history["val_accuracy"],
        label="Validation Accuracy"
    )
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss graph
    axes[1].plot(
        history["epoch"] + 1,
        history["loss"],
        label="Training Loss"
    )
    axes[1].plot(
        history["epoch"] + 1,
        history["val_loss"],
        label="Validation Loss"
    )
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "baseline_training_history.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Training history graphs saved to: {output_path}")

if __name__ == "__main__":
    plot_confusion_matrix()
    plot_training_history()
