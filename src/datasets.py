
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    "data/archive (1)/audio_speech_actors_01-24"
)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(path):
    """Extract emotion and actor ID from a RAVDESS filename."""
    path = Path(path)
    fields = path.stem.split("-")

    if len(fields) != 7:
        return None

    modality, channel, emotion, intensity, statement, repetition, actor = fields

    # Accept audio-only speech recordings only.
    if modality != "03" or channel != "01":
        return None

    if emotion not in EMOTION_MAP:
        return None

    try:
        actor_id = int(actor)
    except ValueError:
        return None

    if not 1 <= actor_id <= 24:
        return None

    return {
        "path": str(path),
        "emotion": EMOTION_MAP[emotion],
        "emotion_id": int(emotion) - 1,
        "actor_id": actor_id,
    }


def load_dataset(dataset_dir=DATASET_DIR):
    """Find valid speech files and extract their labels."""
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir.resolve()}"
        )

    records = []

    for path in sorted(dataset_dir.rglob("*.wav")):
        record = parse_ravdess_filename(path)

        if record is not None:
            records.append(record)

    if not records:
        raise ValueError(
            "No valid speech files found. Check dataset filenames."
        )

    # Detect duplicate paths that refer to the same file location.
    unique_records = {}
    for record in records:
        unique_records[Path(record["path"]).resolve()] = record

    return list(unique_records.values())


if __name__ == "__main__":
    print("Loading RAVDESS dataset...")

    records = load_dataset()

    print(f"\nValid speech files: {len(records)}")
    print(f"Unique actors: {len(set(r['actor_id'] for r in records))}")

    print("\nEmotion distribution:")
    counts = Counter(record["emotion"] for record in records)

    for emotion in EMOTION_MAP.values():
        print(f"{emotion:10s}: {counts.get(emotion, 0)}")

    print("\nFirst 5 records:")
    for record in records[:5]:
        print(record)

    print("\nDataset loading completed.")
