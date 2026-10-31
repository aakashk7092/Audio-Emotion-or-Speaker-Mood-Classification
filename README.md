# Voice Emotion AI

A beginner-friendly Speech Emotion Recognition project. Upload a WAV file or
record with your microphone to estimate the emotion expressed in speech.

## Start here

Open PowerShell in this folder and run:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py audit
python main.py prepare
python main.py train
python main.py evaluate
python main.py app
```

Then open `http://localhost:8501`.

For teacher presentation notes, read [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md).

## Project summary

| Part | Choice |
|---|---|
| Dataset | RAVDESS speech: 1,440 unique clips from 24 actors |
| Classes | Neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Features | 3-Channel Spectrograms (MFCC + Delta + Delta-Delta) + Log-Mel |
| Recommended model | Deep Learning 2D CNN with SpecAugment (or Enhanced SVM) |
| Interface | Modern Glassmorphism Streamlit UI (Full-width, zero sidebar) |
| Evaluation | Strictly Speaker-independent split (4 held-out unseen actors) |

## Verified benchmark

The recommended Deep Learning model was evaluated on 240 clips from four actors not used in
training or validation:

- **Held-out Accuracy: 61.25%** (vs 57.08% SVM baseline)
- **Held-out Macro-F1: 61.15%**
- **Macro Precision: 72.78%**
- High confidence per class: Angry (92.0% Precision), Surprised (90.9% Precision), Happy (93.3% Precision)

This is an experimental result on acted speech. Natural microphone recordings
may be harder because of different speakers, noise, language, and style.

## Simple commands

```powershell
# Check clips, labels, and speaker split
python main.py audit

# Build reusable features and the speaker split (run once after a fresh clone)
python main.py prepare

# Train recommended model
python main.py train

# Evaluate on unseen test speakers
python main.py evaluate

# Predict a WAV file
python main.py predict path\to\recording.wav

# Start web application
python main.py app

# Run automated tests
python main.py test
```

## Easy file map

```text
app.py                  Web interface
main.py                 One-command project runner
src/datasets.py         Reads RAVDESS filenames and labels
src/features.py         Audio preprocessing and features
src/split_data.py       Creates speaker-independent data split
src/train_svm.py        Recommended model training
src/evaluate_svm.py     Fair held-out evaluation
src/predict_svm.py      Prediction for one WAV file
src/audit_dataset.py    Dataset/split validation
PRESENTATION_GUIDE.md   Presentation and viva answers
```

Other `src/train_*.py` files are older or optional experiments. You do not need
them to run or present the core project.

## Dataset note

The archive contains a duplicate RAVDESS folder. The project intentionally uses
only the configured 1,440 unique speech clips. Training with duplicates would
cause leakage and falsely improve accuracy.
