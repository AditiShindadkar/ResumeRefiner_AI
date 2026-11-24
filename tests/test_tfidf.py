from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_refiner.tfidf import score_tfidf

SAMPLES = Path(__file__).parent.parent / "public" / "samples"
RESUME = (SAMPLES / "sample_resume.txt").read_text(encoding="utf-8")
JD = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")


def sklearn_tfidf(a, b):
    try:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


@pytest.mark.parametrize("a, b", [
    (RESUME, JD),
    ("python sql python dashboards", "Python and SQL for dashboards"),
    ("Café résumé naïve", "cafe resume"),
    ("the and of", "a an the"),           # only stop words
    ("C++ C# .NET", "c++ developer"),
    (RESUME, RESUME),
])
def test_matches_sklearn(a, b):
    assert score_tfidf(a, b) == pytest.approx(sklearn_tfidf(a, b), abs=1e-9)


def test_empty_inputs():
    assert score_tfidf("", "x") is None
    assert score_tfidf("x", "") is None
