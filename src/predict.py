
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import EMOTIONS, N_MFCC, PROJECT_ROOT
from src.features import load_audio, extract_mfcc


# Baseline model and its matching normalization.
MODEL_PATH = PROJECT_ROOT / "models" / "best_mfcc_cnn.keras"
NORMALIZATION_PATH = (
    PROJECT_ROOT / "artifacts" / "mfcc_normalization.npz"
)


def load_prediction_resources():
    """Load the trained model and normalization statistics."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(
            f"Normalization file not found: {NORMALIZATION_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with np.load(NORMALIZATION_PATH, allow_pickle=False) as stats:
        mean = float(stats["mean"])
        std = float(stats["std"])

    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
        raise ValueError("Invalid normalization statistics.")

    return model, mean, std


def predict_emotion(
    audio_path: str | Path,
    model=None,
    mean=None,
    std=None,
) -> dict:
    """Predict emotion probabilities for an audio file."""

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Load resources when they were not supplied by the caller.
    if model is None or mean is None or std is None:
        model, mean, std = load_prediction_resources()

    # Reuse the same preprocessing as training.
    audio = load_audio(audio_path)
    mfcc = extract_mfcc(audio)

    expected_shape = (N_MFCC, 130)

    if mfcc.shape != expected_shape:
        raise ValueError(
            f"Expected MFCC shape {expected_shape}, "
            f"received {mfcc.shape}"
        )

    if not np.isfinite(mfcc).all():
        raise ValueError("MFCC features contain invalid values.")

    # Apply baseline training normalization.
    mfcc = (mfcc - mean) / std

    model_input = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

    probabilities = model.predict(model_input, verbose=0)[0]

    if len(probabilities) != len(EMOTIONS):
        raise ValueError(
            f"Expected {len(EMOTIONS)} class probabilities, "
            f"received {len(probabilities)}"
        )

    predicted_id = int(np.argmax(probabilities))

    return {
        "emotion": EMOTIONS[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            emotion: float(probabilities[i])
            for i, emotion in enumerate(EMOTIONS)
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotion from an audio recording."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to an audio file, such as a WAV file",
    )
    args = parser.parse_args()

    try:
        model, mean, std = load_prediction_resources()

        result = predict_emotion(
            args.audio_path,
            model=model,
            mean=mean,
            std=std,
        )

        print(f"\nPredicted emotion: {result['emotion']}")
        print(f"Model confidence: {result['confidence']:.2%}")
        print("\nClass probabilities:")

        for emotion, probability in sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {emotion:10s}: {probability:.2%}")

    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
