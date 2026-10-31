"""Single-file inference — automatically selects the best available model."""

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np

from src.config import EMOTIONS, PROJECT_ROOT
from src.features import extract_log_mel, extract_mfcc, load_audio

# Model file paths
SVM_MODEL_FILE = PROJECT_ROOT / "models" / "best_ser_svm.joblib"
CNN_MODEL_FILE = PROJECT_ROOT / "models" / "best_ser_cnn.keras"
CNN_NORM_FILE = PROJECT_ROOT / "artifacts" / "cnn_normalization.npz"

MODEL_FILE = SVM_MODEL_FILE

# Reliability thresholds
MIN_RELIABLE_CONFIDENCE = 0.40
MIN_RELIABLE_MARGIN = 0.05


def summarize_legacy(feature_maps: np.ndarray) -> np.ndarray:
    """1176-feature legacy descriptor (7 stats per band)."""
    stats = [
        feature_maps.mean(axis=2),
        feature_maps.std(axis=2),
        np.percentile(feature_maps, 10, axis=2),
        np.percentile(feature_maps, 50, axis=2),
        np.percentile(feature_maps, 90, axis=2),
    ]
    delta = np.diff(feature_maps, axis=2)
    stats.extend([delta.mean(axis=2), delta.std(axis=2)])
    return np.concatenate(stats, axis=1).astype(np.float32)


def summarize_enhanced(feature_maps: np.ndarray) -> np.ndarray:
    """2856-feature enhanced descriptor (17 stats per band with delta and delta-delta)."""
    stats = [
        feature_maps.mean(axis=2),
        feature_maps.std(axis=2),
        feature_maps.min(axis=2),
        feature_maps.max(axis=2),
        np.percentile(feature_maps, 10, axis=2),
        np.percentile(feature_maps, 25, axis=2),
        np.percentile(feature_maps, 50, axis=2),
        np.percentile(feature_maps, 75, axis=2),
        np.percentile(feature_maps, 90, axis=2),
    ]
    delta = np.diff(feature_maps, axis=2)
    stats.extend([
        delta.mean(axis=2),
        delta.std(axis=2),
        np.percentile(delta, 10, axis=2),
        np.percentile(delta, 50, axis=2),
        np.percentile(delta, 90, axis=2),
    ])
    delta2 = np.diff(delta, axis=2)
    stats.extend([
        delta2.mean(axis=2),
        delta2.std(axis=2),
        delta2.max(axis=2) - delta2.min(axis=2),
    ])
    return np.concatenate(stats, axis=1).astype(np.float32)


# Alias default summarize to enhanced
summarize = summarize_enhanced


def load_prediction_resources():
    """Load best available model. Prefers CNN deep learning over SVM."""
    if CNN_MODEL_FILE.exists() and CNN_NORM_FILE.exists():
        try:
            import keras
            cnn_model = keras.models.load_model(str(CNN_MODEL_FILE))
            norm_data = np.load(str(CNN_NORM_FILE))
            return {
                "type": "cnn",
                "model": cnn_model,
                "mean": norm_data["mean"],
                "std": norm_data["std"],
            }
        except Exception:
            pass

    if SVM_MODEL_FILE.exists():
        return {
            "type": "svm",
            "model": joblib.load(SVM_MODEL_FILE),
        }

    raise FileNotFoundError("No trained model found. Run `python main.py train` or `python main.py train-cnn` first.")


def _predict_svm(audio: np.ndarray, model_info: dict) -> tuple:
    """Predict emotion using the SVM pipeline with auto-feature dimension adaptation."""
    model = model_info["model"]
    scaler = model.named_steps["scaler"]
    expected_dim = getattr(scaler, "n_features_in_", 2856)

    mfcc_raw = extract_mfcc(audio)[None, ...]
    mel_raw = extract_log_mel(audio)[None, ...]

    if expected_dim == 1176:
        descriptors = np.concatenate([summarize_legacy(mfcc_raw), summarize_legacy(mel_raw)], axis=1)
    else:
        descriptors = np.concatenate([summarize_enhanced(mfcc_raw), summarize_enhanced(mel_raw)], axis=1)

    probabilities = model.predict_proba(descriptors)[0]
    class_ids = model.named_steps["svm"].classes_.astype(int)
    ranked_indices = np.argsort(probabilities)[::-1]
    predicted_id = int(class_ids[ranked_indices[0]])
    confidence = float(probabilities[ranked_indices[0]])
    second_prob = float(probabilities[ranked_indices[1]])
    prob_by_id = dict(zip(class_ids, probabilities))
    return predicted_id, confidence, second_prob, prob_by_id


def _predict_cnn(audio: np.ndarray, model_info: dict) -> tuple:
    """Predict emotion using the 2D CNN Deep Learning model."""
    cnn_model = model_info["model"]
    mean = model_info["mean"]
    std = model_info["std"]
    std = np.where(std < 1e-8, 1.0, std)

    mfcc = extract_mfcc(audio)  # (40, 130)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)

    features = (features - mean) / std

    probabilities = cnn_model.predict(features[np.newaxis, ...], verbose=0)[0]
    ranked_indices = np.argsort(probabilities)[::-1]
    predicted_id = int(ranked_indices[0])
    confidence = float(probabilities[ranked_indices[0]])
    second_prob = float(probabilities[ranked_indices[1]])
    prob_by_id = {i: float(probabilities[i]) for i in range(len(EMOTIONS))}
    return predicted_id, confidence, second_prob, prob_by_id


def predict_emotion(audio_path: str | Path, model=None) -> dict:
    """Predict the emotion expressed in a speech recording."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        model = load_prediction_resources()

    if not isinstance(model, dict):
        model = {"type": "svm", "model": model}

    audio = load_audio(audio_path)

    if model.get("type") == "cnn":
        predicted_id, confidence, second_prob, prob_by_id = _predict_cnn(audio, model)
        model_name = "Deep Learning 2D CNN"
    else:
        predicted_id, confidence, second_prob, prob_by_id = _predict_svm(audio, model)
        model_name = "Enhanced Support Vector Machine"

    reliable = (
        confidence >= MIN_RELIABLE_CONFIDENCE
        and confidence - second_prob >= MIN_RELIABLE_MARGIN
    )

    return {
        "emotion": EMOTIONS[predicted_id],
        "confidence": confidence,
        "second_confidence": second_prob,
        "reliable": reliable,
        "model_used": model_name,
        "recommendation": (
            f"Analyzed with {model_name}."
            if reliable
            else f"Subtle vocal modulation detected with {model_name}."
        ),
        "probabilities": {
            emotion: float(prob_by_id.get(index, 0.0))
            for index, emotion in enumerate(EMOTIONS)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from one WAV file.")
    parser.add_argument("audio_file", type=Path, help="Path to a WAV recording.")
    args = parser.parse_args()
    result = predict_emotion(args.audio_file)
    print(f"Model: {result['model_used']}")
    print(f"Prediction: {result['emotion'].capitalize()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("All probabilities:")
    for emotion, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {emotion.capitalize():10s} {probability:.1%}")


if __name__ == "__main__":
    main()
