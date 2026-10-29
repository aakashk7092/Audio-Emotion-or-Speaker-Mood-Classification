
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.augment import augment_mfcc
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

# Separate files: do not overwrite the baseline experiment.
BEST_MODEL_PATH = MODEL_DIR / "best_mfcc_cnn_augmented.keras"
NORMALIZATION_PATH = (
    ARTIFACT_DIR / "mfcc_normalization_augmented.npz"
)
HISTORY_PATH = ARTIFACT_DIR / "training_history_augmented.csv"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary_augmented.json"


# --------------------------------------------------
# 3. Augmented training batches
# --------------------------------------------------
class AugmentedMFCCSequence(tf.keras.utils.Sequence):
    """Apply MFCC feature augmentation to training batches only."""

    def __init__(
        self,
        features,
        labels,
        batch_size=32,
        noise_level=0.03,
        max_shift=8,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(labels))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        batch_indices = self.indices[start:start + self.batch_size]

        # Copy so the original normalized features remain unchanged.
        batch_x = self.features[batch_indices].copy()
        batch_y = self.labels[batch_indices]

        for i in range(len(batch_x)):
            # Apply augmentation to 70% of samples.
            if self.rng.random() < 0.70:
                # CNN input: (MFCC, time, channel)
                mfcc_map = batch_x[i, :, :, 0]

                batch_x[i, :, :, 0] = augment_mfcc(
                    mfcc_map,
                    noise_level=self.noise_level,
                    max_shift=self.max_shift,
                    rng=self.rng,
                )

        return batch_x, batch_y

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)


# --------------------------------------------------
# 4. Train model
# --------------------------------------------------
def train_model():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    if not SPLIT_FILE.exists():
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

    # Actor-independent split.
    train_idx = np.flatnonzero(np.isin(actor_ids, train_actors))
    val_idx = np.flatnonzero(np.isin(actor_ids, val_actors))
    test_idx = np.flatnonzero(np.isin(actor_ids, test_actors))

    assert set(train_actors).isdisjoint(val_actors)
    assert set(train_actors).isdisjoint(test_actors)
    assert set(val_actors).isdisjoint(test_actors)

    assert (
        len(train_idx) + len(val_idx) + len(test_idx)
        == len(labels)
    )

    print(f"Training samples:   {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples:       {len(test_idx)}")

    # CNN input shape: (samples, 40, 130, 1)
    X_train = mfcc[train_idx][..., np.newaxis]
    X_val = mfcc[val_idx][..., np.newaxis]

    y_train = labels[train_idx]
    y_val = labels[val_idx]

    # Normalize using training data only.
    mean = float(X_train.mean())
    std = float(X_train.std())

    if std < 1e-8:
        raise ValueError("MFCC standard deviation is too small.")

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    np.savez(
        NORMALIZATION_PATH,
        mean=np.array(mean, dtype=np.float32),
        std=np.array(std, dtype=np.float32),
    )

    print(f"MFCC mean: {mean:.5f}")
    print(f"MFCC std:  {std:.5f}")
    print(f"Saved normalization: {NORMALIZATION_PATH}")

    # Class weights are computed from original training labels.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print(f"Class weights: {class_weights}")

    train_sequence = AugmentedMFCCSequence(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        noise_level=0.03,
        max_shift=8,
        seed=RANDOM_SEED,
    )

    # Quick batch sanity check before expensive training.
    sample_x, sample_y = train_sequence[0]
    assert sample_x.shape[1:] == (40, 130, 1)
    assert len(sample_x) == len(sample_y)
    assert np.isfinite(sample_x).all()

    print("Augmentation batch test passed.")
    print(f"First batch shape: {sample_x.shape}")

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

    print("\nStarting augmented CNN training...", flush=True)

    history = model.fit(
        train_sequence,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    summary = {
        "experiment": "MFCC feature-level augmentation",
        "epochs_completed": len(history.history["loss"]),
        "best_val_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_val_loss": float(
            min(history.history["val_loss"])
        ),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples_reserved": int(len(test_idx)),
        "augmentation_probability": 0.70,
        "noise_level": 0.03,
        "max_time_shift_frames": 8,
        "best_model_path": str(BEST_MODEL_PATH),
        "normalization_path": str(NORMALIZATION_PATH),
        "history_path": str(HISTORY_PATH),
        "test_used_for_model_selection": False,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nAugmented training completed.")
    print(f"Epochs completed: {summary['epochs_completed']}")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"New model: {BEST_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("The test set was not used for training or model selection.")


if __name__ == "__main__":
    train_model()
