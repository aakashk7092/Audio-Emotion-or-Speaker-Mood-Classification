"""SpecAugment-style data augmentation for speech emotion recognition."""

import numpy as np


def frequency_mask(features, max_width=5, num_masks=2):
    """Zero out random horizontal bands in a spectrogram."""
    augmented = features.copy()
    n_freq = augmented.shape[0]
    for _ in range(num_masks):
        width = np.random.randint(1, max_width + 1)
        start = np.random.randint(0, max(1, n_freq - width))
        augmented[start : start + width, :] = 0.0
    return augmented


def time_mask(features, max_width=12, num_masks=2):
    """Zero out random vertical bands in a spectrogram."""
    augmented = features.copy()
    n_time = augmented.shape[1]
    for _ in range(num_masks):
        width = np.random.randint(1, max_width + 1)
        start = np.random.randint(0, max(1, n_time - width))
        augmented[:, start : start + width] = 0.0
    return augmented


def add_noise(features, scale=0.015):
    """Add small Gaussian noise to a spectrogram."""
    return features + np.random.normal(0, scale, features.shape).astype(np.float32)


def time_shift(features, max_shift=15):
    """Circular-shift along the time axis."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(features, shift, axis=1)


def spec_augment(features):
    """Apply a random combination of SpecAugment transforms to one sample.

    *features* has shape ``(freq, time)`` or ``(freq, time, channels)``.
    """
    is_3d = features.ndim == 3
    if is_3d:
        channels = [features[:, :, c] for c in range(features.shape[2])]
    else:
        channels = [features]

    augmented = []
    for ch in channels:
        out = ch
        if np.random.random() < 0.7:
            out = frequency_mask(out, max_width=4, num_masks=2)
        if np.random.random() < 0.7:
            out = time_mask(out, max_width=10, num_masks=2)
        if np.random.random() < 0.5:
            out = add_noise(out, scale=0.01)
        if np.random.random() < 0.3:
            out = time_shift(out, max_shift=10)
        augmented.append(out)

    if is_3d:
        return np.stack(augmented, axis=-1).astype(np.float32)
    return augmented[0].astype(np.float32)


def augment_dataset(X, y, num_augments=4):
    """Create *num_augments* augmented copies of every sample in *X*.

    Parameters
    ----------
    X : ndarray, shape ``(N, freq, time)`` or ``(N, freq, time, channels)``
    y : ndarray, shape ``(N,)``
    num_augments : int

    Returns
    -------
    X_aug, y_aug : ndarrays with the original data followed by augmented copies.
    """
    batches_X = [X]
    batches_y = [y]
    for _ in range(num_augments):
        aug = np.array([spec_augment(sample) for sample in X], dtype=np.float32)
        batches_X.append(aug)
        batches_y.append(y.copy())
    return np.concatenate(batches_X), np.concatenate(batches_y)

