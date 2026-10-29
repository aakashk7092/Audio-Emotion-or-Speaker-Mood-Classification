
import json
from pathlib import Path

import librosa
import numpy as np

from src.config import (
    DATASET_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_AUDIO,
    N_MFCC,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    EMOTIONS,
)
from src.datasets import load_dataset


def load_audio(audio_path):
    """Load audio, trim silence, and standardize its duration."""
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    # Trim leading and trailing silence.
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if audio.size == 0:
        raise ValueError(f"No usable audio found: {audio_path}")

    # Crop or zero-pad to the configured duration.
    if len(audio) > SAMPLES_PER_AUDIO:
        audio = audio[:SAMPLES_PER_AUDIO]
    elif len(audio) < SAMPLES_PER_AUDIO:
        audio = np.pad(
            audio,
            (0, SAMPLES_PER_AUDIO - len(audio)),
            mode="constant",
        )

    return audio.astype(np.float32)


def extract_mfcc(audio):
    """Extract MFCC features with shape (N_MFCC, time_frames)."""
    features = librosa.feature.mfcc(
        y=np.asarray(audio, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    return features.astype(np.float32)


def extract_log_mel(audio):
    """Extract a log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.astype(np.float32)


def extract_features(audio_path):
    """Load one audio file and extract MFCC and log-Mel features."""
    audio = load_audio(audio_path)

    mfcc = extract_mfcc(audio)
    log_mel = extract_log_mel(audio)

    return mfcc, log_mel


def build_feature_dataset():
    """Extract and save features for the complete dataset."""
    print("Loading dataset...", flush=True)

    records = load_dataset(DATASET_DIR)
    total = len(records)

    if total == 0:
        raise RuntimeError(f"No audio files found in {DATASET_DIR}")

    print(f"Found {total} audio files.", flush=True)
    print("Starting feature extraction...", flush=True)

    mfcc_list = []
    log_mel_list = []
    labels = []
    actor_ids = []
    paths = []
    failures = []

    for index, record in enumerate(records, start=1):
        try:
            mfcc, log_mel = extract_features(record["path"])

            if not (
                np.isfinite(mfcc).all()
                and np.isfinite(log_mel).all()
            ):
                raise ValueError("Non-finite feature values detected")

            mfcc_list.append(mfcc)
            log_mel_list.append(log_mel)
            labels.append(record["emotion_id"])
            actor_ids.append(record["actor_id"])
            paths.append(str(record["path"]))

        except Exception as error:
            failures.append({
                "path": str(record["path"]),
                "error": str(error),
            })

        if index % 25 == 0 or index == total:
            print(
                f"Processed {index}/{total} | "
                f"Successful: {len(labels)} | "
                f"Failed: {len(failures)}",
                flush=True,
            )

    if not labels:
        raise RuntimeError(
            "No audio files were processed successfully."
        )

    # Ensure every extracted feature has the expected shape.
    expected_mfcc_shape = (N_MFCC, 130)
    expected_mel_shape = (N_MELS, 130)

    for features in mfcc_list:
        if features.shape != expected_mfcc_shape:
            raise ValueError(
                f"Unexpected MFCC shape: {features.shape}"
            )

    for features in log_mel_list:
        if features.shape != expected_mel_shape:
            raise ValueError(
                f"Unexpected log-Mel shape: {features.shape}"
            )

    mfcc_array = np.stack(mfcc_list).astype(np.float32)
    log_mel_array = np.stack(log_mel_list).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    actors_array = np.asarray(actor_ids, dtype=np.int16)
    paths_array = np.asarray(paths, dtype=str)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DIR / "features_dataset.npz"
    failure_file = PROCESSED_DIR / "feature_extraction_errors.json"

    print("\nSaving extracted features...", flush=True)

    np.savez_compressed(
        output_file,
        mfcc=mfcc_array,
        log_mel=log_mel_array,
        labels=labels_array,
        actor_ids=actors_array,
        paths=paths_array,
        emotions=np.asarray(EMOTIONS, dtype=str),
    )

    with failure_file.open("w", encoding="utf-8") as file:
        json.dump(failures, file, indent=2)

    print("\nFeature extraction completed.", flush=True)
    print(f"Saved file: {output_file}", flush=True)
    print(f"Successful samples: {len(labels)}", flush=True)
    print(f"Failed samples: {len(failures)}", flush=True)
    print(f"MFCC shape: {mfcc_array.shape}", flush=True)
    print(f"Log-Mel shape: {log_mel_array.shape}", flush=True)
    print(f"Labels shape: {labels_array.shape}", flush=True)
    print(f"Actor IDs shape: {actors_array.shape}", flush=True)
    print(f"Error report: {failure_file}", flush=True)


if __name__ == "__main__":
    build_feature_dataset()
