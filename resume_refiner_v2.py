"""
ResumeRefiner AI — v2
=====================

Changes over v1:

1. CHUNKED EMBEDDING
   v1 encoded the entire resume as one string. all-MiniLM-L6-v2 silently
   truncates input at 256 word pieces, so a 2-page resume (~900 tokens) was
   represented almost entirely by its header and first job. v2 splits both
   documents into token-budgeted chunks, embeds each chunk, and aggregates
   with a JD-coverage score (for each JD requirement, how well is it covered
   anywhere in the resume?).

2. WORD-BOUNDARY SKILL MATCHING
   v1 used `skill.lower() in text.lower()`. That reports "Java" as present
   when the resume only says "JavaScript", and "R" as present in every word
   containing an r. v2 uses boundary-aware regex plus an alias table so
   "sklearn", "scikit learn" and "Scikit-Learn" all resolve to one skill.

3. EVALUATION HARNESS
   v1 had no ground truth and no metrics. v2 adds an Evaluate tab that takes
   a labelled CSV of (resume, JD, human_score) triples and reports Spearman /
   Pearson / ROC-AUC for three scorers side by side:
       - TF-IDF cosine       (lexical baseline)
       - whole-doc embedding  (v1 behaviour, truncated)
       - chunked embedding    (v2)
   This is what turns the project from a demo into something defensible.

Run:  streamlit run resume_refiner_v2.py
"""

import io
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional / heavy imports guarded so the app degrades instead of crashing --
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import mammoth
except ImportError:
    mammoth = None

try:
    import ollama
except ImportError:
    ollama = None

import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr


# ============================================================================
# Configuration
# ============================================================================

OLLAMA_MODEL = "mistral"
NLP_MODEL = "en_core_web_sm"
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates at 256 word pieces. We budget 200 to leave room
# for special tokens and to avoid landing exactly on the boundary.
MAX_CHUNK_TOKENS = 200
MODEL_HARD_LIMIT = 256


# ---------------------------------------------------------------------------
# Skill vocabulary: canonical name -> surface forms seen in real documents.
# Expanding this is the cheapest accuracy win in the whole app.
# ---------------------------------------------------------------------------
SKILL_ALIASES = {
    # Languages
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "JavaScript": ["javascript", "java script", "js", "es6"],
    "TypeScript": ["typescript"],
    "R": ["r programming", "r language"],  # bare "R" is too noisy to match
    "SQL": ["sql"],
    "NoSQL": ["nosql"],
    "Scala": ["scala"],
    "Go": ["golang"],

    # Data / BI
    "Power BI": ["power bi", "powerbi"],
    "DAX": ["dax"],
    "Power Query": ["power query"],
    "Tableau": ["tableau"],
    "Looker": ["looker"],
    "Excel": ["excel", "ms excel", "microsoft excel"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],

    # ML / DS
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Statistics": ["statistics", "statistical analysis"],
    "A/B Testing": ["a/b testing", "ab testing", "hypothesis testing"],
    "Time Series": ["time series", "forecasting"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "TensorFlow": ["tensorflow", "tensor flow"],
    "PyTorch": ["pytorch", "py torch"],
    "Keras": ["keras"],
    "XGBoost": ["xgboost", "xg boost"],
    "LightGBM": ["lightgbm", "light gbm"],
    "SHAP": ["shap"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "LLM": ["llm", "large language model", "large language models"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],

    # Engineering / infra
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Git": ["git", "github", "version control"],
    "Linux": ["linux", "unix"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi", "fast api"],
    "Streamlit": ["streamlit"],
    "REST API": ["rest api", "restful", "rest apis"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs", "node js"],

    # Soft / process
    "Project Management": ["project management"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Stakeholder Management": ["stakeholder management", "stakeholder"],
    "Communication": ["communication"],
    "Leadership": ["leadership"],
    "Teamwork": ["teamwork", "collaboration", "cross-functional"],
}


# ============================================================================
# FIX 2 — boundary-aware skill matching
# ============================================================================

def _compile(surface: str) -> re.Pattern:
    """
    Build a boundary-aware pattern for one surface form.

    We can't use \\b because it fails on tokens ending in non-word characters:
    \\bC\\+\\+\\b never matches "C++" (there is no word boundary after '+').
    Instead we assert that the character on each side is not part of a
    programming-identifier alphabet.

        "java"  vs "JavaScript"  -> blocked, next char 'S' is alphabetic
        "c++"   vs "C++"         -> matches
        "ml"    vs "html"        -> blocked, previous char 't' is alphabetic
    """
    edge = r"[A-Za-z0-9+#_]"
    return re.compile(rf"(?<!{edge}){re.escape(surface)}(?!{edge})", re.IGNORECASE)


COMPILED_SKILLS = {
    canonical: [_compile(s) for s in surfaces]
    for canonical, surfaces in SKILL_ALIASES.items()
}


def find_skills(text: str) -> set:
    """Return the set of canonical skill names present in `text`."""
    if not text:
        return set()
    return {
        canonical
        for canonical, patterns in COMPILED_SKILLS.items()
        if any(p.search(text) for p in patterns)
    }


def compare_skills(resume_text: str, jd_text: str):
    """Skills the JD asks for, split into present-in-resume vs missing."""
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)
    present = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    return present, missing, extra


# ============================================================================
# Model loading
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(
            f"spaCy model '{model_name}' not found. Install it with:\n\n"
            f"`python -m spacy download {model_name}`"
        )
        return None


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_name: str):
    return SentenceTransformer(model_name)


nlp = load_spacy_model(NLP_MODEL)
st_model = load_sentence_transformer(SENTENCE_MODEL)


def count_tokens(text: str) -> int:
    """Exact word-piece count using the model's own tokenizer."""
    return len(st_model.tokenizer.encode(text, add_special_tokens=False))


# ============================================================================
# FIX 1 — chunking so the whole document is actually read
# ============================================================================

def split_into_units(text: str) -> list:
    """
    Break text into the smallest sensible units.

    Resumes are bullet-driven, so line breaks carry more structure than
    sentence boundaries. We split on newlines first, and only fall back to
    spaCy sentence segmentation for lines that are still too long.
    """
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if count_tokens(line) <= MAX_CHUNK_TOKENS:
            units.append(line)
        elif nlp is not None:
            doc = nlp(line)
            for sent in doc.sents:
                s = sent.text.strip()
                if s:
                    units.append(s)
        else:
            units.append(line)
    return units


def pack_units(units: list, max_tokens: int = MAX_CHUNK_TOKENS) -> list:
    """
    Greedily pack units into chunks that stay under the token budget.

    Packing (rather than one-chunk-per-line) matters: a lone bullet like
    "Python, SQL, Power BI" embeds poorly in isolation, while a packed chunk
    of related lines carries enough context for a meaningful vector.
    """
    chunks, current, current_tokens = [], [], 0

    for unit in units:
        n = count_tokens(unit)

        # A single unit that still exceeds the budget: hard-split on words.
        if n > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            words = unit.split()
            step = max(1, int(max_tokens * 0.6))  # ~0.6 words per word-piece
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + step]))
            continue

        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0

        current.append(unit)
        current_tokens += n

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def chunk_text(text: str) -> list:
    """Full chunking pipeline. Guaranteed: every chunk fits the model."""
    if not text or not text.strip():
        return []
    return pack_units(split_into_units(text))


@st.cache_data(show_spinner=False)
def embed_chunks(chunks: tuple):
    """Embed a tuple of chunks. Tuple (not list) so Streamlit can hash it."""
    if not chunks:
        return None
    return st_model.encode(list(chunks), convert_to_tensor=True,
                           normalize_embeddings=True)


def score_chunked(resume_text: str, jd_text: str) -> dict:
    """
    JD-coverage score.

    For every JD chunk we ask: "is this requirement addressed anywhere in the
    resume?" — i.e. take the MAX similarity across all resume chunks. The
    overall score is the mean of those maxima.

    Max-then-mean is the right aggregation here. Mean-then-mean would punish a
    candidate for having relevant experience the JD didn't ask about, which is
    not what we want to measure.
    """
    resume_chunks = chunk_text(resume_text)
    jd_chunks = chunk_text(jd_text)

    if not resume_chunks or not jd_chunks:
        return {"score": None, "detail": pd.DataFrame(),
                "n_resume_chunks": 0, "n_jd_chunks": 0}

    r_emb = embed_chunks(tuple(resume_chunks))
    j_emb = embed_chunks(tuple(jd_chunks))

    sim = util.cos_sim(j_emb, r_emb).cpu().numpy()   # (n_jd, n_resume)
    best_idx = sim.argmax(axis=1)
    best_val = sim.max(axis=1)

    detail = pd.DataFrame({
        "JD requirement": [jd_chunks[i][:180] for i in range(len(jd_chunks))],
        "Coverage": best_val,
        "Best matching resume section": [resume_chunks[j][:180] for j in best_idx],
    }).sort_values("Coverage")

    return {
        "score": float(best_val.mean()),
        "detail": detail,
        "n_resume_chunks": len(resume_chunks),
        "n_jd_chunks": len(jd_chunks),
    }


def score_whole_doc(resume_text: str, jd_text: str):
    """v1 behaviour, kept only so the evaluation tab can prove chunking wins."""
    if not resume_text or not jd_text:
        return None
    emb = st_model.encode([resume_text, jd_text], convert_to_tensor=True,
                          normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def score_tfidf(resume_text: str, jd_text: str):
    """Lexical baseline. Always report a baseline — it is the first question."""
    if not resume_text or not jd_text:
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        m = vec.fit_transform([resume_text, jd_text])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


# ============================================================================
# Text extraction
# ============================================================================

def extract_text_from_pdf(uploaded_file):
    if pdfplumber is None:
        st.error("pdfplumber is not installed: `pip install pdfplumber`")
        return None
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [p.extract_text() for p in pdf.pages]
        text = "\n".join(p for p in pages if p)
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return None


def extract_text_from_docx(uploaded_file):
    if mammoth is None:
        st.error("mammoth is not installed: `pip install mammoth`")
        return None
    try:
        text = mammoth.extract_raw_text(uploaded_file).value.strip()
        return text or None
    except Exception as e:
        st.error(f"Could not read DOCX: {e}")
        return None


def check_grammar(text: str) -> list:
    """Single heuristic: plural subject with a singular verb."""
    if nlp is None:
        return []
    doc = nlp(text)
    return [
        f"Possible subject-verb agreement issue: '{t.text}' → '{t.head.text}'"
        for t in doc
        if t.dep_ == "nsubj" and t.tag_ in ("NNS", "NNPS") and t.head.tag_ == "VBZ"
    ]


def analyze_with_ollama(resume_text: str, jd_text: str, missing_skills: list):
    if ollama is None:
        return None
    prompt = f"""You are an experienced technical recruiter.
Review this resume against the job description and give specific, actionable feedback.

Cover: keyword alignment, quantified impact, clarity, and overall fit.
Be concrete — quote the exact bullet you would change and show the rewrite.

Our automated analysis found these required skills absent from the resume:
{', '.join(missing_skills) if missing_skills else 'none'}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""
    try:
        r = ollama.chat(model=OLLAMA_MODEL,
                        messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"]
    except Exception as e:
        st.warning(f"Ollama unavailable ({e}). Skipping LLM feedback.")
        return None


# ============================================================================
# FIX 3 — evaluation harness
# ============================================================================

def evaluate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a labelled dataset with all three scorers and report agreement
    with human labels.

    Expected columns: resume_text, jd_text, label
    `label` may be an ordinal fit rating (e.g. 1-5) or binary (0/1).
    """
    rows = []
    progress = st.progress(0.0, text="Scoring pairs...")

    for i, row in enumerate(df.itertuples(index=False)):
        rows.append({
            "label": row.label,
            "tfidf": score_tfidf(row.resume_text, row.jd_text),
            "whole_doc": score_whole_doc(row.resume_text, row.jd_text),
            "chunked": score_chunked(row.resume_text, row.jd_text)["score"],
        })
        progress.progress((i + 1) / len(df), text=f"Scored {i + 1}/{len(df)} pairs")

    progress.empty()
    scored = pd.DataFrame(rows).dropna()

    labels = scored["label"].to_numpy(dtype=float)
    is_binary = set(np.unique(labels)) <= {0.0, 1.0}

    results = []
    for name, col in [("TF-IDF baseline", "tfidf"),
                      ("Whole-doc embedding (v1)", "whole_doc"),
                      ("Chunked embedding (v2)", "chunked")]:
        preds = scored[col].to_numpy(dtype=float)
        rho, p_rho = spearmanr(labels, preds)
        r, _ = pearsonr(labels, preds)
        auc = roc_auc_score(labels, preds) if is_binary else np.nan
        results.append({
            "Scorer": name,
            "Spearman rho": round(rho, 4),
            "p-value": f"{p_rho:.4f}",
            "Pearson r": round(r, 4),
            "ROC-AUC": round(auc, 4) if is_binary else "n/a",
        })

    st.session_state["eval_raw"] = scored
    return pd.DataFrame(results)


def make_template_csv() -> bytes:
    template = pd.DataFrame({
        "resume_text": ["Paste plain-text resume here", "Another resume"],
        "jd_text": ["Paste the job description here", "Another JD"],
        "label": [3, 1],
    })
    buf = io.StringIO()
    template.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ============================================================================
# UI
# ============================================================================

st.set_page_config(page_title="ResumeRefiner AI v2", layout="wide")
st.title("ResumeRefiner AI")
st.caption("Chunked semantic matching · boundary-aware skill extraction · evaluated against human labels")

tab_analyze, tab_eval, tab_about = st.tabs(["Analyze", "Evaluate", "How it works"])


# ------------------------------- Analyze ------------------------------------
with tab_analyze:
    left, right = st.columns([1, 2])

    with left:
        uploaded = st.file_uploader("Resume (PDF or DOCX)", type=["pdf", "docx"])
        jd_text = st.text_area("Job description", height=260,
                               placeholder="Paste the full job description...")
        use_llm = st.checkbox("Include local LLM feedback (requires Ollama)",
                              value=False)
        run = st.button("Analyze", type="primary", use_container_width=True)

    resume_text = None
    if uploaded is not None:
        if uploaded.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded)
        else:
            resume_text = extract_text_from_docx(uploaded)

    with right:
        if resume_text:
            resume_text = st.text_area("Extracted text (editable)", resume_text,
                                       height=260, key="resume_area")
            st.caption(
                f"{count_tokens(resume_text)} word pieces · "
                f"model limit is {MODEL_HARD_LIMIT} per encode, so this "
                f"document needs {len(chunk_text(resume_text))} chunks."
            )
        else:
            st.info("Upload a resume to begin.")

    if run and resume_text and jd_text:
        with st.spinner("Analyzing..."):
            result = score_chunked(resume_text, jd_text)
            present, missing, extra = compare_skills(resume_text, jd_text)
            grammar = check_grammar(resume_text)
            baseline = score_tfidf(resume_text, jd_text)
            v1_score = score_whole_doc(resume_text, jd_text)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("JD coverage (chunked)", f"{result['score']:.1%}")
        c2.metric("Whole-doc score (v1)", f"{v1_score:.1%}",
                  help="What the old truncated approach would have returned.")
        c3.metric("TF-IDF baseline", f"{baseline:.1%}")
        st.caption(
            f"Resume split into {result['n_resume_chunks']} chunks, "
            f"JD into {result['n_jd_chunks']}. Scores are only meaningful "
            "relative to other candidates — MiniLM cosine values rarely "
            "approach 0, so don't read 45% as 'failing'."
        )

        st.subheader("Weakest-covered requirements")
        st.dataframe(result["detail"].head(10), use_container_width=True,
                     hide_index=True)

        s1, s2 = st.columns(2)
        with s1:
            st.subheader("Skills")
            if present:
                st.success("**Matched:** " + ", ".join(present))
            if missing:
                st.warning("**Required but absent:** " + ", ".join(missing))
            if extra:
                st.info("**In resume, not requested:** " + ", ".join(extra))
        with s2:
            st.subheader("Grammar")
            if grammar:
                for g in grammar:
                    st.write("- " + g)
            else:
                st.success("No subject-verb agreement issues detected.")
            st.caption("Single-rule heuristic — not a full grammar checker.")

        if use_llm:
            with st.spinner("Querying local LLM..."):
                feedback = analyze_with_ollama(resume_text, jd_text, missing)
            if feedback:
                st.subheader("LLM feedback")
                st.markdown(feedback)


# ------------------------------- Evaluate -----------------------------------
with tab_eval:
    st.markdown(
        "Upload a labelled CSV with columns **resume_text**, **jd_text**, "
        "**label**. Label can be an ordinal fit rating (1–5) or binary (0/1). "
        "Thirty to fifty hand-labelled pairs is enough to say something honest."
    )
    st.download_button("Download CSV template", make_template_csv(),
                       "eval_template.csv", "text/csv")

    eval_file = st.file_uploader("Labelled CSV", type=["csv"], key="eval_upload")

    if eval_file is not None:
        df = pd.read_csv(eval_file)
        required = {"resume_text", "jd_text", "label"}
        if not required.issubset(df.columns):
            st.error(f"CSV must contain: {', '.join(sorted(required))}")
        else:
            df = df.dropna(subset=["resume_text", "jd_text", "label"])
            st.write(f"Loaded {len(df)} labelled pairs.")
            if st.button("Run evaluation", type="primary"):
                results = evaluate_dataset(df)
                st.subheader("Agreement with human labels")
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.caption(
                    "Spearman is the headline metric: it measures whether the "
                    "scorer *ranks* candidates the way a human would, which is "
                    "what a screening tool is actually for."
                )
                st.subheader("Raw scores")
                st.dataframe(st.session_state["eval_raw"],
                             use_container_width=True, hide_index=True)


# ------------------------------- About --------------------------------------
with tab_about:
    st.markdown(f"""
### Why chunking

`{SENTENCE_MODEL}` truncates input at {MODEL_HARD_LIMIT} word pieces. A
two-page resume is roughly 900. Encoding it as one string means the vector
describes the header and first job, and nothing else — quietly, with no
warning. Every long resume then lands in a narrow score band, because the
model is largely comparing contact details.

v2 splits on line breaks (resumes are bullet-structured), falls back to spaCy
sentence segmentation for overlong lines, and greedily packs units into
{MAX_CHUNK_TOKENS}-token chunks.

Aggregation is **max-then-mean**: for each JD requirement take the best-matching
resume chunk, then average. This measures requirement coverage. Mean-then-mean
would penalise a candidate for having relevant experience the JD didn't ask
about.

### Why regex over substring matching

`"java" in "javascript"` is `True`. v1 therefore credited candidates with Java
whenever they listed JavaScript. v2 asserts that neighbouring characters are
outside `[A-Za-z0-9+#_]`. Plain `\\b` doesn't work — there is no word boundary
after the `+` in `C++`.

### Why evaluation

Three scorers run over the same labelled pairs: TF-IDF (lexical baseline),
whole-doc embedding (v1), chunked embedding (v2). Without the baseline you
cannot claim the transformer earned its place; without v1 you cannot claim
chunking helped.

### Known limitations

- Skill list is curated, so unlisted skills are invisible. A trained NER model
  would replace this.
- Grammar check is one rule.
- Cosine scores are not calibrated probabilities — use them to rank, not to
  set a hiring threshold.
- No bias audit. Names, universities and gendered language remain in the text
  and can influence embeddings. Strip PII before any real deployment.
""")
