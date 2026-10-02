import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

MODELS = ("nb", "logreg")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"£|\$|€", " money ", text)
    # номера телефонов и короткие номера для ответа - частый признак спама
    text = re.sub(r"\d{5,}", " longnum ", text)
    return text


def build(kind: str = "nb", min_df: int = 2) -> Pipeline:
    vectorizer = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                                 min_df=min_df, sublinear_tf=True)
    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=10, class_weight="balanced")
    else:
        raise ValueError(f"неизвестная модель: {kind}, есть {MODELS}")
    return make_pipeline(vectorizer, clf)


def evaluate(model: Pipeline, texts, labels) -> dict:
    pred = model.predict(texts)
    return {
        "f1": f1_score(labels, pred),
        "report": classification_report(labels, pred, target_names=["ham", "spam"], digits=3),
        "confusion": confusion_matrix(labels, pred).tolist(),
    }
