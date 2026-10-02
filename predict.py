import argparse
import sys

import joblib

from train import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка сообщений на спам")
    parser.add_argument("texts", nargs="*", help="сообщения; если пусто, читаются строки из stdin")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("Модель не найдена, сначала запустите python train.py")
    model = joblib.load(MODEL_PATH)

    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    for text, proba in zip(texts, model.predict_proba(texts)[:, 1]):
        label = "SPAM" if proba >= 0.5 else "ok  "
        print(f"{label} {proba:.2f}  {text[:80]}")


if __name__ == "__main__":
    main()
