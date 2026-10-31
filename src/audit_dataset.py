"""Validate dataset integrity, label balance, and speaker-disjoint partitions."""

import json
from collections import Counter

import numpy as np
import soundfile as sf

from src.config import EMOTIONS, PROCESSED_DIR
from src.datasets import load_dataset


def main():
    records = load_dataset()
    corrupt = []
    for record in records:
        try:
            info = sf.info(record["path"])
            if info.frames <= 0 or info.samplerate <= 0:
                corrupt.append(record["path"])
        except RuntimeError:
            corrupt.append(record["path"])

    labels = Counter(record["emotion"] for record in records)
    actors = Counter(record["actor_id"] for record in records)
    expected_labels = {"neutral": 96, **{emotion: 192 for emotion in EMOTIONS[1:]}}
    if len(records) != 1440 or labels != expected_labels or len(actors) != 24 or corrupt:
        raise RuntimeError(
            f"Dataset audit failed: clips={len(records)}, actors={len(actors)}, corrupt={len(corrupt)}, labels={dict(labels)}"
        )

    with (PROCESSED_DIR / "actor_splits.json").open(encoding="utf-8") as stream:
        split = json.load(stream)
    train, validation, test = (set(split[key]) for key in ("train_actors", "validation_actors", "test_actors"))
    if train & validation or train & test or validation & test or train | validation | test != set(actors):
        raise RuntimeError("Actor split has overlap or does not cover all 24 speakers.")

    with np.load(PROCESSED_DIR / "features_dataset.npz", allow_pickle=False) as data:
        if len(data["labels"]) != len(records):
            raise RuntimeError("Feature cache and source dataset have different clip counts.")
    print(json.dumps({
        "status": "passed",
        "unique_ravdess_clips": len(records),
        "actors": len(actors),
        "label_counts": dict(labels),
        "speaker_split": {key: split[key] for key in ("train_actors", "validation_actors", "test_actors")},
        "corrupt_wav_files": len(corrupt),
    }, indent=2))


if __name__ == "__main__":
    main()
