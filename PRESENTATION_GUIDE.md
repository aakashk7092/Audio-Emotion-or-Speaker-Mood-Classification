# Teacher Presentation Guide

## Project in one sentence

This project predicts the **expressed emotion in a speech recording** using
audio features and a machine-learning classifier, then shows the result in a
web application with upload and microphone recording options.

## Dataset

- **Dataset:** RAVDESS speech audio
- **Unique audio clips used:** 1,440
- **Speakers:** 24 actors
- **Emotion classes:** neutral, calm, happy, sad, angry, fearful, disgust,
  surprised
- **Important:** The archive contains a duplicate folder. The pipeline uses
  only the 1,440 unique clips, so no duplicate samples leak into training.

## Workflow

```text
WAV speech input
    -> silence trim + fixed 3-second audio
    -> MFCC + log-Mel acoustic features
    -> statistical feature summaries
    -> StandardScaler + RBF SVM
    -> emotion probabilities + top prediction
```

## Fair evaluation

We split by **speaker**, not randomly by clip: 16 training speakers, 4
validation speakers, and 4 unseen test speakers (240 recordings). This avoids
memorising a speaker's voice.

## Current verified result

Recommended deployed model: **Deep Learning 2D CNN with SpecAugment Data Augmentation**.

- **Held-out Accuracy: 61.25%** (Increased from 57.08%)
- **Held-out Macro-F1: 61.15%**
- **Macro Precision: 72.78%**
- Outstanding per-class results: Angry (92% Precision, 81% F1), Surprised (91% Precision, 74% F1), Happy (93% Precision), Calm (74% Precision, 80% F1).
- Training accuracy reached **94.2%** with early stopping preventing overfit.

Baseline comparison:
- Classical SVM Baseline: 57.08% Acc / 56.39% F1
- Enhanced SVM (2856 stats): 59.17% Acc / 58.71% F1
- **Deep 2D-CNN (SpecAugment + 3D Tensor)**: **61.25% Acc / 61.15% F1** (Best)

## Demo steps

1. Run `python main.py app`.
2. Open `http://localhost:8501`.
3. Record 1–5 seconds or upload a WAV file.
4. Click **Analyze emotion**.
5. Select **New recording** to try another clip.

## Viva answers

**Why MFCC and log-Mel?** They describe speech frequencies and vocal-tract
information useful for emotion recognition.

**Why speaker-independent split?** It tests voices the model never saw during
training.

**Why not 95% accuracy?** Emotion is subjective, RAVDESS has only 24 acted
speakers, and microphone speech differs from studio recordings. Honest
held-out evaluation is better than duplicate-data leakage.

**Future improvement?** Larger diverse datasets and a pretrained Wav2Vec2
model. The project contains an optional advanced Wav2Vec2 experiment, but the
app uses it only if it beats the verified baseline.
