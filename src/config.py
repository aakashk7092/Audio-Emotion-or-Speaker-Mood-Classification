
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset configuration
DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "archive (1)"
    / "audio_speech_actors_01-24"
)

# Processed data directory
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Audio configuration
SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES_PER_AUDIO = int(SAMPLE_RATE * DURATION)

# Feature configuration
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Reproducibility
RANDOM_SEED = 42

# Emotion labels in numerical order
EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]
