
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from src.config import PROJECT_ROOT, EMOTIONS, RANDOM_SEED
from src.models import build_mfcc_cnn_regularized


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# 2. Load feature dataset
# --------------------------------------------------

data_path = PROJECT_ROOT / "data" / "processed" / "features_dataset.npz"
split_path = PROJECT_ROOT / "data" / "processed" / "actor_splits.json"

data = np.load(data_path, allow_pickle=True)

print("Available dataset keys:", data.files)

X = data["mfcc"].astype(np.float32)
y = data["labels"].astype(np.int64)


# --------------------------------------------------
# 3. Get actor IDs
# --------------------------------------------------

if "actors" in data.files:
    actors = data["actors"].astype(np.int64)

elif "actor_ids" in data.files:
    actors = data["actor_ids"].astype(np.int64)

elif "paths" in data.files:
    actors = np.array([
        int(Path(str(path)).stem.split("-")[-1])
        for path in data["paths"]
    ], dtype=np.int64)

else:
    raise KeyError(
        "Actor information not found. "
        f"Available keys: {data.files}"
    )

assert len(X) == len(y) == len(actors), (
    "Mismatch between features, labels and actor IDs."
)


# --------------------------------------------------
# 4. Load actor-independent split
# --------------------------------------------------

with open(split_path, "r") as file:
    splits = json.load(file)

train_actors = splits["train_actors"]
val_actors = splits["validation_actors"]
test_actors = splits["test_actors"]

train_mask = np.isin(actors, train_actors)
val_mask = np.isin(actors, val_actors)
test_mask = np.isin(actors, test_actors)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

assert set(actors[train_mask]).isdisjoint(set(actors[val_mask]))
assert set(actors[train_mask]).isdisjoint(set(actors[test_mask]))
assert set(actors[val_mask]).isdisjoint(set(actors[test_mask]))

assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0

print("\nDataset loaded successfully")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test reserved:", X_test.shape)
print("Training actors:", sorted(set(actors[train_mask])))
print("Validation actors:", sorted(set(actors[val_mask])))
print("Test actors:", sorted(set(actors[test_mask])))


# --------------------------------------------------
# 5. Normalize using training data only
# --------------------------------------------------

mean = float(X_train.mean())
std = float(X_train.std())

X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("\nNormalization mean:", mean)
print("Normalization std:", std)


# --------------------------------------------------
# 6. Class weights
# --------------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# --------------------------------------------------
# 7. Build Experiment 4B model
# --------------------------------------------------

model = build_mfcc_cnn_regularized(
    input_shape=(40, 130, 1),
    num_classes=len(EMOTIONS),
    dropout_rates=(0.15, 0.20, 0.30),
    l2_strength=0.0,
    learning_rate=0.001,
)

model.summary()


# --------------------------------------------------
# 8. Output paths
# --------------------------------------------------

models_dir = PROJECT_ROOT / "models"
artifacts_dir = PROJECT_ROOT / "artifacts"

models_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = models_dir / "best_mfcc_cnn_reg_4b.keras"
history_path = artifacts_dir / "training_history_reg_4b.csv"
normalization_path = artifacts_dir / "mfcc_normalization_reg_4b.npz"
summary_path = artifacts_dir / "training_summary_reg_4b.json"


# --------------------------------------------------
# 9. Callbacks
# --------------------------------------------------

callbacks = [
    ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),

    CSVLogger(str(history_path)),
]


# --------------------------------------------------
# 10. Train
# --------------------------------------------------

print("\nStarting Experiment 4B...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# --------------------------------------------------
# 11. Save normalization
# --------------------------------------------------

np.savez(
    normalization_path,
    mean=mean,
    std=std,
)


# --------------------------------------------------
# 12. Save experiment summary
# --------------------------------------------------

history_df = pd.DataFrame(history.history)

best_epoch = int(history_df["val_accuracy"].idxmax())
best_val_accuracy = float(history_df["val_accuracy"].max())
best_val_loss = float(history_df["val_loss"].min())

summary = {
    "experiment": "4B",
    "dropout_rates": [0.15, 0.20, 0.30],
    "l2_strength": 0.0,
    "learning_rate": 0.001,
    "epochs_completed": len(history_df),
    "best_epoch": best_epoch + 1,
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss,
    "test_samples_reserved": int(len(y_test)),
    "best_model_path": str(checkpoint_path),
    "normalization_path": str(normalization_path),
}

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

print("\nExperiment 4B completed!")
print("Best epoch:", best_epoch + 1)
print("Best validation accuracy:", best_val_accuracy)
print("Best validation loss:", best_val_loss)
print("Model saved:", checkpoint_path)
print("Summary saved:", summary_path)
print("Test dataset was NOT used for evaluation.")
