"""
Token-budgeted chunking so the whole document is read by the encoder.

all-MiniLM-L6-v2 silently truncates at 256 word pieces; a two-page resume is
~900. These functions take the token counter as an argument so they can be
tested without downloading a model.
"""

from typing import Callable, Iterable, List, Optional

from .config import MAX_CHUNK_TOKENS

TokenCounter = Callable[[str], int]
Sentencizer = Callable[[str], Iterable[str]]


def split_into_units(text: str, count_tokens: TokenCounter,
                     max_tokens: int = MAX_CHUNK_TOKENS,
                     sentencizer: Optional[Sentencizer] = None) -> List[str]:
    """
    Break text into the smallest sensible units.

    Resumes are bullet-driven, so line breaks carry more structure than
    sentence boundaries. Split on newlines first and only fall back to
    sentence segmentation for lines that are still too long.
    """
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if count_tokens(line) <= max_tokens or sentencizer is None:
            units.append(line)
        else:
            units.extend(s.strip() for s in sentencizer(line) if s.strip())
    return units


def pack_units(units: List[str], count_tokens: TokenCounter,
               max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Greedily pack units into chunks under the token budget.

    A lone bullet like "Python, SQL, Power BI" embeds poorly on its own; a
    packed chunk of neighbouring lines carries enough context to be useful.
    """
    chunks, current, current_tokens = [], [], 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(current))
        current, current_tokens = [], 0

    for unit in units:
        n = count_tokens(unit)

        if n > max_tokens:
            # Still too long after sentence splitting: hard-split on words,
            # shrinking the window until every piece fits.
            flush()
            words = unit.split()
            step = max(1, int(max_tokens * 0.6))
            i = 0
            while i < len(words):
                size = step
                piece = " ".join(words[i:i + size])
                while size > 1 and count_tokens(piece) > max_tokens:
                    size = max(1, size // 2)
                    piece = " ".join(words[i:i + size])
                chunks.append(piece)
                i += size
            continue

        if current and current_tokens + n > max_tokens:
            flush()
        current.append(unit)
        current_tokens += n

    flush()
    return [c for c in chunks if c.strip()]


def split_requirements(text: str, count_tokens: TokenCounter,
                       max_tokens: int = MAX_CHUNK_TOKENS,
                       sentencizer: Optional[Sentencizer] = None,
                       min_words: int = 4) -> List[str]:
    """
    Split a job description into individual requirements: one per line or
    sentence, without packing, so each requirement is scored on its own.
    Heading-like lines shorter than `min_words` are dropped.
    """
    if not text or not text.strip():
        return []
    units = split_into_units(text, count_tokens, max_tokens, sentencizer)
    reqs = [u for u in units if len(u.split()) >= min_words]
    # Pack each requirement on its own; pack_units hard-splits any that
    # are still over budget.
    return [c for u in reqs for c in pack_units([u], count_tokens, max_tokens)]


def chunk_text(text: str, count_tokens: TokenCounter,
               max_tokens: int = MAX_CHUNK_TOKENS,
               sentencizer: Optional[Sentencizer] = None) -> List[str]:
    """Full chunking pipeline: every returned chunk fits within `max_tokens`."""
    if not text or not text.strip():
        return []
    units = split_into_units(text, count_tokens, max_tokens, sentencizer)
    return pack_units(units, count_tokens, max_tokens)
