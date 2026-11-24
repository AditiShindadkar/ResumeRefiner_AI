"""Compare scorers against human fit labels."""

import io
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from .scoring import EmbeddingScorer, score_tfidf

REQUIRED_COLUMNS = {"resume_text", "jd_text", "label"}

SCORERS = [
    ("TF-IDF baseline", "tfidf"),
    ("Whole-doc embedding (v1)", "whole_doc"),
    ("Chunked embedding (v2)", "chunked"),
]


def score_dataset(df: pd.DataFrame, scorer: EmbeddingScorer,
                  on_progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        rows.append({
            "label": row.label,
            "tfidf": score_tfidf(row.resume_text, row.jd_text),
            "whole_doc": scorer.score_whole_doc(row.resume_text, row.jd_text),
            "chunked": scorer.score_chunked(row.resume_text, row.jd_text).score,
        })
        if on_progress:
            on_progress(i + 1, len(df))
    return pd.DataFrame(rows).dropna()


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Spearman / Pearson / ROC-AUC of each scorer against the labels."""
    labels = scored["label"].to_numpy(dtype=float)
    is_binary = set(np.unique(labels)) <= {0.0, 1.0}
    enough = len(scored) >= 3 and len(np.unique(labels)) > 1

    results = []
    for name, col in SCORERS:
        preds = scored[col].to_numpy(dtype=float)
        if enough and np.ptp(preds) > 0:
            rho, p_rho = spearmanr(labels, preds)
            r, _ = pearsonr(labels, preds)
            auc = roc_auc_score(labels, preds) if is_binary else None
        else:
            rho = p_rho = r = auc = None
        results.append({
            "Scorer": name,
            "Spearman rho": None if rho is None else round(float(rho), 4),
            "p-value": None if p_rho is None else round(float(p_rho), 4),
            "Pearson r": None if r is None else round(float(r), 4),
            "ROC-AUC": None if auc is None else round(float(auc), 4),
        })
    return pd.DataFrame(results)


def evaluate(df: pd.DataFrame, scorer: EmbeddingScorer,
             on_progress=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored = score_dataset(df, scorer, on_progress)
    return summarize(scored), scored


def template_csv() -> bytes:
    template = pd.DataFrame({
        "resume_text": ["Paste plain-text resume here", "Another resume"],
        "jd_text": ["Paste the job description here", "Another JD"],
        "label": [3, 1],
    })
    buf = io.StringIO()
    template.to_csv(buf, index=False)
    return buf.getvalue().encode()
