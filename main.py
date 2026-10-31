"""One-command entry point for the beginner-friendly SER project."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_module(module: str, extra: list[str] | None = None) -> None:
    subprocess.run([sys.executable, "-m", module, *(extra or [])], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition: simple project commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Extract features and create the speaker split.")
    commands.add_parser("audit", help="Check dataset files, labels and speaker split.")
    commands.add_parser("train", help="Train the enhanced SVM model.")
    commands.add_parser("train-cnn", help="Train the CNN deep-learning model.")
    commands.add_parser("train-all", help="Train both SVM and CNN models.")
    commands.add_parser("evaluate", help="Evaluate SVM on unseen speakers.")
    commands.add_parser("evaluate-cnn", help="Evaluate CNN on unseen speakers.")
    commands.add_parser("evaluate-all", help="Evaluate both SVM and CNN models.")
    predict = commands.add_parser("predict", help="Predict one WAV file.")
    predict.add_argument("audio_file", help="Path to a WAV file.")
    commands.add_parser("app", help="Start the Streamlit app.")
    commands.add_parser("test", help="Run automated tests.")
    args = parser.parse_args()
    actions = {
        "prepare": lambda: (run_module("src.features"), run_module("src.split_data")),
        "audit": lambda: run_module("src.audit_dataset"),
        "train": lambda: run_module("src.train_svm"),
        "train-cnn": lambda: run_module("src.train_cnn"),
        "train-all": lambda: (run_module("src.train_svm"), run_module("src.train_cnn")),
        "evaluate": lambda: run_module("src.evaluate_svm"),
        "evaluate-cnn": lambda: run_module("src.evaluate_cnn"),
        "evaluate-all": lambda: (run_module("src.evaluate_svm"), run_module("src.evaluate_cnn")),
        "predict": lambda: run_module("src.predict_svm", [args.audio_file]),
        "test": lambda: subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True),
        "app": lambda: subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT, check=True),
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
