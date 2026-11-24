import numpy as np
import pandas as pd

from resume_refiner.evaluation import summarize
from resume_refiner.scoring import EmbeddingScorer, score_tfidf


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False, **_):
        return text.split()


class FakeModel:
    """Bag-of-words 'embedding' over a tiny vocabulary, L2-normalised."""
    vocab = ["python", "sql", "dashboard", "cooking", "forecast"]
    tokenizer = FakeTokenizer()

    def encode(self, texts, normalize_embeddings=True, **_):
        out = []
        for t in texts:
            words = t.lower().split()
            v = np.array([words.count(w) for w in self.vocab], dtype=float) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_tfidf_handles_empty_and_identical():
    assert score_tfidf("", "x") is None
    assert score_tfidf("python sql", "python sql") > 0.99


def test_chunked_coverage_prefers_relevant_resume():
    scorer = EmbeddingScorer(model=FakeModel(), max_tokens=5)
    jd = "python sql\ndashboard forecast"
    good = scorer.score_chunked("python sql dashboard\nforecast", jd)
    bad = scorer.score_chunked("cooking cooking\ncooking", jd)
    assert good.score > bad.score
    assert good.n_jd_chunks >= 1 and not good.detail.empty


def test_chunked_empty_inputs():
    scorer = EmbeddingScorer(model=FakeModel())
    assert scorer.score_chunked("", "python").score is None


def test_summarize_handles_constant_predictions():
    scored = pd.DataFrame({"label": [0, 1, 0, 1], "tfidf": [0.1, 0.9, 0.2, 0.8],
                           "whole_doc": [0.5] * 4, "chunked": [0.3, 0.7, 0.2, 0.9]})
    out = summarize(scored).set_index("Scorer")
    assert out.loc["TF-IDF baseline", "ROC-AUC"] == 1.0
    assert pd.isna(out.loc["Whole-doc embedding (v1)", "Spearman rho"])
