
import json

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_SEED


def create_actor_splits():
    feature_file = PROCESSED_DIR / "features_dataset.npz"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {feature_file}"
        )

    print("Loading extracted features...", flush=True)

    with np.load(feature_file, allow_pickle=False) as data:
        labels = data["labels"]
        actor_ids = data["actor_ids"]

    unique_actors = np.unique(actor_ids)

    if len(unique_actors) != 24:
        raise ValueError(
            f"Expected 24 actors, found {len(unique_actors)}"
        )

    # First reserve 4 actors for testing.
    train_val_actors, test_actors = train_test_split(
        unique_actors,
        test_size=4,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Reserve 4 of the remaining actors for validation.
    train_actors, val_actors = train_test_split(
        train_val_actors,
        test_size=4,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_indices = np.flatnonzero(
        np.isin(actor_ids, train_actors)
    )
    val_indices = np.flatnonzero(
        np.isin(actor_ids, val_actors)
    )
    test_indices = np.flatnonzero(
        np.isin(actor_ids, test_actors)
    )

    # Verify no actor is shared across splits.
    assert set(train_actors).isdisjoint(val_actors)
    assert set(train_actors).isdisjoint(test_actors)
    assert set(val_actors).isdisjoint(test_actors)

    # Verify every sample belongs to exactly one split.
    assert (
        len(train_indices)
        + len(val_indices)
        + len(test_indices)
        == len(labels)
    )

    split_info = {
        "seed": RANDOM_SEED,
        "train_actors": sorted(map(int, train_actors)),
        "validation_actors": sorted(map(int, val_actors)),
        "test_actors": sorted(map(int, test_actors)),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
    }

    output_file = PROCESSED_DIR / "actor_splits.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2)

    print("\nActor-independent split created successfully.")
    print(f"Training actors:   {split_info['train_actors']}")
    print(f"Validation actors: {split_info['validation_actors']}")
    print(f"Test actors:       {split_info['test_actors']}")

    print(f"\nTraining samples:   {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Testing samples:    {len(test_indices)}")

    print(f"\nSaved split: {output_file}")
    print("Actor leakage check: PASSED")


if __name__ == "__main__":
    create_actor_splits()
