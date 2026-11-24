"""Three scorers: TF-IDF baseline, whole-document embedding, chunked coverage."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .chunking import chunk_text, split_requirements
from .config import (MAX_CHUNK_TOKENS, MIN_REQUIREMENT_WORDS,
                     RESUME_CHUNK_TOKENS, SENTENCE_MODEL)
from .tfidf import score_tfidf  # noqa: F401  (re-exported)


@dataclass
class CoverageResult:
    score: Optional[float]
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_resume_chunks: int = 0
    n_jd_chunks: int = 0


class EmbeddingScorer:
    """
    Wraps a sentence-transformer model (and optional spaCy pipeline used for
    sentence splitting). Embeddings are cached per chunk string, so
    re-analysing an edited resume only re-encodes the lines that changed.
    """

    def __init__(self, model=None, nlp=None, max_tokens: int = MAX_CHUNK_TOKENS,
                 resume_chunk_tokens: int = RESUME_CHUNK_TOKENS):
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SENTENCE_MODEL)
        self.model = model
        self.nlp = nlp
        self.max_tokens = max_tokens
        self.resume_chunk_tokens = min(resume_chunk_tokens, max_tokens)
        self._cache = {}

    # -- helpers ------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        # verbose=False: counting a long document is fine; we never feed it to the model unchunked.
        return len(self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def _sentences(self, text: str):
        return (s.text for s in self.nlp(text).sents)

    def _sentencizer(self):
        return self._sentences if self.nlp is not None else None

    def chunk(self, text: str):
        """Resume chunks: neighbouring lines packed to `resume_chunk_tokens`."""
        return chunk_text(text, self.count_tokens, self.resume_chunk_tokens,
                          self._sentencizer())

    def requirements(self, jd_text: str):
        """JD units: one requirement per line/sentence. Falls back to packed
        chunks if the JD is a single block with no line structure."""
        reqs = split_requirements(jd_text, self.count_tokens, self.max_tokens,
                                  self._sentencizer(), MIN_REQUIREMENT_WORDS)
        return reqs or chunk_text(jd_text, self.count_tokens, self.max_tokens,
                                  self._sentencizer())

    def embed(self, texts):
        """Return an (n, d) numpy array of L2-normalised embeddings."""
        import numpy as np

        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self.model.encode(todo, normalize_embeddings=True)
            self._cache.update(zip(todo, vecs))
        return np.vstack([self._cache[t] for t in texts])

    # -- scorers ------------------------------------------------------------
    def score_whole_doc(self, resume_text: str, jd_text: str) -> Optional[float]:
        """v1 behaviour: one vector per document (truncated by the model)."""
        if not resume_text or not jd_text:
            return None
        emb = self.model.encode([resume_text, jd_text], normalize_embeddings=True)
        return float(emb[0] @ emb[1])

    def score_chunked(self, resume_text: str, jd_text: str) -> CoverageResult:
        """
        JD-coverage score, aggregated max-then-mean.

        For each JD requirement take its best-matching resume chunk, then
        average those maxima. Mean-then-mean would penalise relevant experience
        the JD didn't ask about, which isn't what we want to measure.
        """
        resume_chunks = self.chunk(resume_text or "")
        jd_chunks = self.requirements(jd_text or "")
        if not resume_chunks or not jd_chunks:
            return CoverageResult(score=None)

        sim = self.embed(jd_chunks) @ self.embed(resume_chunks).T  # (n_jd, n_resume)
        best_idx = sim.argmax(axis=1)
        best_val = sim.max(axis=1)

        detail = pd.DataFrame({
            "JD requirement": [c[:180] for c in jd_chunks],
            "Coverage": best_val,
            "Best matching resume section": [resume_chunks[j][:180] for j in best_idx],
        }).sort_values("Coverage")

        return CoverageResult(
            score=float(best_val.mean()),
            detail=detail,
            n_resume_chunks=len(resume_chunks),
            n_jd_chunks=len(jd_chunks),
        )
