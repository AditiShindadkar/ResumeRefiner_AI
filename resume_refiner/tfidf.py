"""
Pure-Python TF-IDF cosine similarity.

Matches scikit-learn's TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
followed by cosine similarity, without pulling scikit-learn/scipy (~150 MB)
into the serverless bundle. tests/test_tfidf.py checks parity with sklearn.
"""

import math
import re
from collections import Counter
from typing import List, Optional

# scikit-learn's ENGLISH_STOP_WORDS (BSD-3-Clause), copied verbatim.
STOP_WORDS = frozenset("""
a about above across after afterwards again against all almost alone along already also
although always am among amongst amoungst amount an and another any anyhow anyone
anything anyway anywhere are around as at back be became because become becomes becoming
been before beforehand behind being below beside besides between beyond bill both bottom
but by call can cannot cant co con could couldnt cry de describe detail do done down due
during each eg eight either eleven else elsewhere empty enough etc even ever every
everyone everything everywhere except few fifteen fifty fill find fire first five for
former formerly forty found four from front full further get give go had has hasnt have
he hence her here hereafter hereby herein hereupon hers herself him himself his how
however hundred i ie if in inc indeed interest into is it its itself keep last latter
latterly least less ltd made many may me meanwhile might mill mine more moreover most
mostly move much must my myself name namely neither never nevertheless next nine no
nobody none noone nor not nothing now nowhere of off often on once one only onto or
other others otherwise our ours ourselves out over own part per perhaps please put
rather re same see seem seemed seeming seems serious several she should show side since
sincere six sixty so some somehow someone something sometime sometimes somewhere still
such system take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they thick thin third this those though three
through throughout thru thus to together too top toward towards twelve twenty two un
under until up upon us very via was we well were what whatever when whence whenever
where whereafter whereas whereby wherein whereupon wherever whether which while whither
who whoever whole whom whose why will with within without would yet you your yours
yourself yourselves
""".split())

TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _ngrams(text: str) -> List[str]:
    # sklearn drops stop words *before* forming bigrams.
    tokens = [t for t in TOKEN.findall(text.lower()) if t not in STOP_WORDS]
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def score_tfidf(resume_text: str, jd_text: str) -> Optional[float]:
    """Lexical baseline: cosine similarity of TF-IDF (1-2 gram) vectors."""
    if not resume_text or not jd_text:
        return None
    counts = [Counter(_ngrams(resume_text)), Counter(_ngrams(jd_text))]
    if not (counts[0] or counts[1]):
        return 0.0  # empty vocabulary, e.g. only stop words

    n_docs = 2
    vocab = set(counts[0]) | set(counts[1])
    # Smoothed idf, as in sklearn: ln((1 + n) / (1 + df)) + 1
    idf = {t: math.log((1 + n_docs) / (1 + sum(t in c for c in counts))) + 1
           for t in vocab}

    vecs = []
    for c in counts:
        v = {t: n * idf[t] for t, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        vecs.append({t: x / norm for t, x in v.items()} if norm else {})
    return float(sum(x * vecs[1].get(t, 0.0) for t, x in vecs[0].items()))
