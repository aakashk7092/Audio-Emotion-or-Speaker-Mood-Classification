
import numpy as np


def add_gaussian_noise(
    features: np.ndarray,
    noise_level: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add light Gaussian noise to MFCC features."""
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=features.shape,
    )

    return (features + noise).astype(np.float32)


def shift_time(
    features: np.ndarray,
    max_shift: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift a 2D feature map along its time axis."""
    if features.ndim != 2:
        raise ValueError("Expected a 2D feature array: (MFCC, time).")

    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    shift = int(rng.integers(-max_shift, max_shift + 1))

    if shift == 0:
        return features.copy()

    shifted = np.zeros_like(features)

    if shift > 0:
        shifted[:, shift:] = features[:, :-shift]
    else:
        shifted[:, :shift] = features[:, -shift:]

    return shifted


def augment_mfcc(
    features: np.ndarray,
    noise_level: float = 0.01,
    max_shift: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply feature-level noise and time-shift augmentation."""
    if rng is None:
        rng = np.random.default_rng()

    shifted = shift_time(
        features,
        max_shift=max_shift,
        rng=rng,
    )

    augmented = add_gaussian_noise(
        shifted,
        noise_level=noise_level,
        rng=rng,
    )

    return augmented
