"""Train a CNN on MFCC spectrograms for speaker-independent SER."""

import json

import keras
import librosa
import numpy as np
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

from src.augment import augment_dataset
from src.config import EMOTIONS, PROCESSED_DIR, PROJECT_ROOT, RANDOM_SEED

FEATURE_FILE = PROCESSED_DIR / "features_dataset.npz"
SPLIT_FILE = PROCESSED_DIR / "actor_splits.json"
MODEL_FILE = PROJECT_ROOT / "models" / "best_ser_cnn.keras"
NORM_FILE = PROJECT_ROOT / "artifacts" / "cnn_normalization.npz"
SUMMARY_FILE = PROJECT_ROOT / "artifacts" / "training_summary_cnn.json"
HISTORY_FILE = PROJECT_ROOT / "artifacts" / "training_history_cnn.csv"

NUM_CLASSES = len(EMOTIONS)


# ── Feature preparation ──────────────────────────────────────────────────────

def prepare_cnn_input(mfcc: np.ndarray) -> np.ndarray:
    """Stack static MFCC, delta, and delta-delta into a 3-channel tensor.

    Parameters
    ----------
    mfcc : ndarray, shape ``(N, n_mfcc, time)``

    Returns
    -------
    ndarray, shape ``(N, n_mfcc, time, 3)``
    """
    deltas, delta2s = [], []
    for sample in mfcc:
        deltas.append(librosa.feature.delta(sample, order=1))
        delta2s.append(librosa.feature.delta(sample, order=2))
    delta_arr = np.array(deltas, dtype=np.float32)
    delta2_arr = np.array(delta2s, dtype=np.float32)
    return np.stack([mfcc, delta_arr, delta2_arr], axis=-1).astype(np.float32)


def normalize(X_train, X_val, X_test=None):
    """Global per-channel zero-mean unit-variance normalization."""
    # Compute stats only from training data — shape (3,)
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    if X_test is not None:
        X_test = (X_test - mean) / std
        return X_train, X_val, X_test, mean, std
    return X_train, X_val, mean, std


# ── Model architecture ───────────────────────────────────────────────────────

def build_model(input_shape, num_classes=NUM_CLASSES):
    """Compact CNN with batch-normalization and dropout."""
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)

    # Classifier head
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="ser_cnn")


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    if not FEATURE_FILE.exists() or not SPLIT_FILE.exists():
        raise FileNotFoundError(
            "Run `python main.py prepare` first to extract features and create splits."
        )

    # Load pre-extracted features
    with np.load(FEATURE_FILE, allow_pickle=False) as data:
        mfcc_all = data["mfcc"]  # (N, 40, 130)
        labels_all = data["labels"].astype(np.int64)
        actor_ids = data["actor_ids"].astype(np.int64)

    with SPLIT_FILE.open(encoding="utf-8") as fh:
        split = json.load(fh)

    train_mask = np.isin(actor_ids, split["train_actors"])
    val_mask = np.isin(actor_ids, split["validation_actors"])
    test_mask = np.isin(actor_ids, split["test_actors"])

    # Create 3-channel input: MFCC + delta + delta-delta
    print("Computing delta features...", flush=True)
    X_all = prepare_cnn_input(mfcc_all)  # (N, 40, 130, 3)

    X_train_raw = X_all[train_mask]
    y_train = labels_all[train_mask]
    X_val = X_all[val_mask]
    y_val = labels_all[val_mask]

    # Normalize (fit on training set only)
    print("Normalizing features...", flush=True)
    X_train_norm, X_val_norm, mean, std = normalize(X_train_raw, X_val)

    # Data augmentation on training set
    print("Augmenting training data (4× copies)...", flush=True)
    X_train_aug, y_train_aug = augment_dataset(X_train_norm, y_train, num_augments=4)
    print(f"Training samples after augmentation: {len(X_train_aug)}", flush=True)

    # Balanced class weights
    unique_labels = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=unique_labels, y=y_train_aug)
    class_weight = dict(zip(unique_labels.tolist(), weights.tolist()))
    print(f"Class weights: { {EMOTIONS[k]: round(v, 2) for k, v in class_weight.items()} }", flush=True)

    # Build model
    input_shape = X_train_aug.shape[1:]  # (40, 130, 3)
    model = build_model(input_shape, NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    np.random.seed(RANDOM_SEED)
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    print("\nStarting CNN training...\n", flush=True)
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val_norm, y_val),
        epochs=150,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model and normalization parameters
    MODEL_FILE.parent.mkdir(exist_ok=True)
    NORM_FILE.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_FILE))
    np.savez(str(NORM_FILE), mean=mean, std=std)

    # Save training history
    import pandas as pd
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(str(HISTORY_FILE), index=False)

    # Validation metrics at best epoch
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    summary = {
        "model": "CNN with MFCC + delta + delta-delta (3-channel)",
        "input_shape": list(input_shape),
        "total_params": int(model.count_params()),
        "best_epoch": best_epoch,
        "epochs_trained": len(history.history["loss"]),
        "best_val_accuracy": float(best_val_acc),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "train_samples_original": int(train_mask.sum()),
        "train_samples_augmented": len(X_train_aug),
        "validation_samples": int(val_mask.sum()),
        "test_samples_reserved": int(test_mask.sum()),
        "augmentation": "SpecAugment (freq mask, time mask, noise, time shift) × 4",
        "model_path": str(MODEL_FILE),
    }
    with SUMMARY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"CNN training complete!")
    print(f"Best validation accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Model saved: {MODEL_FILE}")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

