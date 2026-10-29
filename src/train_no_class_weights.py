
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import PROCESSED_DIR, PROJECT_ROOT, RANDOM_SEED
from src.models import build_mfcc_cnn


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------------------------------------------
# 2. Training configuration
# --------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 32

FEATURE_FILE = PROCESSED_DIR / "features_dataset.npz"
SPLIT_FILE = PROCESSED_DIR / "actor_splits.json"

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# All outputs are separate from baseline and augmented experiments.
BEST_MODEL_PATH = MODEL_DIR / "best_mfcc_cnn_no_class_weights.keras"
NORMALIZATION_PATH = (
    ARTIFACT_DIR / "mfcc_normalization_no_class_weights.npz"
)
HISTORY_PATH = ARTIFACT_DIR / "training_history_no_class_weights.csv"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary_no_class_weights.json"


def train_model():
    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature dataset...", flush=True)

    with np.load(FEATURE_FILE, allow_pickle=False) as data:
        mfcc = data["mfcc"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        actor_ids = data["actor_ids"].astype(np.int64)

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        split_info = json.load(file)

    train_actors = split_info["train_actors"]
    val_actors = split_info["validation_actors"]
    test_actors = split_info["test_actors"]

    # Actor-independent split: no actor overlap.
    assert set(train_actors).isdisjoint(val_actors)
    assert set(train_actors).isdisjoint(test_actors)
    assert set(val_actors).isdisjoint(test_actors)

    train_idx = np.flatnonzero(np.isin(actor_ids, train_actors))
    val_idx = np.flatnonzero(np.isin(actor_ids, val_actors))
    test_idx = np.flatnonzero(np.isin(actor_ids, test_actors))

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(labels)

    print(f"Training samples:   {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples reserved: {len(test_idx)}")

    X_train = mfcc[train_idx][..., np.newaxis]
    X_val = mfcc[val_idx][..., np.newaxis]

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # Normalize using training data only.
    mean = float(X_train.mean())
    std = float(X_train.std())

    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
        raise ValueError("Invalid MFCC normalization statistics.")

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    np.savez(
        NORMALIZATION_PATH,
        mean=np.array(mean, dtype=np.float32),
        std=np.array(std, dtype=np.float32),
    )

    print(f"MFCC mean: {mean:.5f}")
    print(f"MFCC std:  {std:.5f}")
    print("Augmentation: disabled")
    print("Class weights: disabled")

    # Regular NumPy batches: no feature augmentation.
    model = build_mfcc_cnn(
        input_shape=(40, 130, 1),
        num_classes=8,
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(
            filename=str(HISTORY_PATH),
            append=False,
        ),
    ]

    print("\nStarting CNN training without class weights or augmentation...",
          flush=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    summary = {
        "experiment": "MFCC CNN without class weights or augmentation",
        "epochs_completed": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples_reserved": int(len(test_idx)),
        "class_weights_used": False,
        "augmentation_used": False,
        "best_model_path": str(BEST_MODEL_PATH),
        "normalization_path": str(NORMALIZATION_PATH),
        "history_path": str(HISTORY_PATH),
        "test_used_for_model_selection": False,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nExperiment completed.")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"Model saved: {BEST_MODEL_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print("Test set was not used for training or model selection.")


if __name__ == "__main__":
    train_model()
