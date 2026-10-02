import pytest

import data
from model import build, evaluate, normalize

SPAM = [
    "WINNER! You have won a free prize, call 09061701461 now",
    "Free entry to win cash prize, text WIN to 87121",
    "Congratulations, you won a free ticket! Call 08002986030",
    "URGENT! Claim your free cash prize now, visit www.win-now.com",
]
HAM = [
    "Are we still meeting for lunch today?",
    "I'll call you when I get home",
    "Can you pick up some milk on the way?",
    "See you at the lecture tomorrow",
]


def test_normalize_replaces_urls_and_numbers():
    text = normalize("Visit http://spam.com or call 0906170146 for £100")
    assert "url" in text and "longnum" in text and "money" in text
    assert "http" not in text


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        build("svm")


@pytest.mark.parametrize("kind", ["nb", "logreg"])
def test_model_separates_obvious_cases(kind):
    model = build(kind, min_df=1).fit(SPAM + HAM, [1] * len(SPAM) + [0] * len(HAM))
    assert model.predict(["You won a free cash prize, call 09061701999"])[0] == 1
    assert model.predict(["Are we meeting at home tomorrow?"])[0] == 0


def test_evaluate_returns_metrics():
    model = build("nb", min_df=1).fit(SPAM + HAM, [1] * 4 + [0] * 4)
    result = evaluate(model, SPAM + HAM, [1] * 4 + [0] * 4)
    assert result["f1"] == 1.0
    assert result["confusion"] == [[4, 0], [0, 4]]


def test_load_parses_tsv(tmp_path):
    path = tmp_path / "sms.tsv"
    path.write_text('ham\tOk "see" you\nspam\tFree prize\nham\tOk "see" you\n', encoding="utf-8")
    df = data.load(path)
    assert list(df.label) == [0, 1]
    assert df.text[0] == 'Ok "see" you'
