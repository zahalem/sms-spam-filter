import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

import data
from model import MODELS, build, evaluate

MODEL_PATH = Path(__file__).parent / "models" / "spam.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение фильтра спама")
    parser.add_argument("--model", choices=[*MODELS, "both"], default="both")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = data.load(data.download())
    print(f"Сообщений: {len(df)}, спама: {df.label.mean():.1%}")
    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=args.test_size, stratify=df.label, random_state=42)

    kinds = MODELS if args.model == "both" else (args.model,)
    best, best_f1 = None, -1.0
    for kind in kinds:
        model = build(kind).fit(x_train, y_train)
        result = evaluate(model, x_test, y_test)
        print(f"\n== {kind}: F1 = {result['f1']:.3f}")
        print(result["report"])
        print("confusion matrix:", result["confusion"])
        if result["f1"] > best_f1:
            best, best_f1 = (kind, model), result["f1"]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best[1], MODEL_PATH)
    print(f"\nСохранена модель {best[0]} (F1 {best_f1:.3f}) в {MODEL_PATH}")


if __name__ == "__main__":
    main()
