"""
ResumeRefiner AI: Streamlit front end.

Run:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from resume_refiner import __version__
from resume_refiner.config import (MAX_CHUNK_TOKENS, MODEL_HARD_LIMIT, NLP_MODEL,
                                   OLLAMA_MODEL, SENTENCE_MODEL)
from resume_refiner.evaluation import REQUIRED_COLUMNS, evaluate, template_csv
from resume_refiner.extract import ExtractionError, extract_text
from resume_refiner.llm import LLMUnavailable, analyze_with_ollama
from resume_refiner.quality import analyze_quality, check_grammar
from resume_refiner.report import build_markdown_report
from resume_refiner.scoring import EmbeddingScorer, score_tfidf
from resume_refiner.skills import compare_skills

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="ResumeRefiner AI", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# Models (loaded once per server process)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading language models...")
def load_models():
    import spacy
    try:
        nlp = spacy.load(NLP_MODEL)
    except OSError:
        nlp = None
    return EmbeddingScorer(nlp=nlp), nlp


scorer, nlp = load_models()
if nlp is None:
    st.warning(
        f"spaCy model `{NLP_MODEL}` not found, so grammar checks and sentence "
        f"splitting are disabled. Install it with `python -m spacy download {NLP_MODEL}`."
    )


# ----------------------------------------------------------------------------
# Session state helpers
# ----------------------------------------------------------------------------

for key in ("resume_text", "jd_text"):
    st.session_state.setdefault(key, "")
st.session_state.setdefault("last_upload", None)


def load_sample():
    st.session_state.resume_text = (DATA_DIR / "sample_resume.txt").read_text(encoding="utf-8")
    st.session_state.jd_text = (DATA_DIR / "sample_jd.txt").read_text(encoding="utf-8")
    st.session_state.pop("analysis", None)


def clear_inputs():
    st.session_state.resume_text = ""
    st.session_state.jd_text = ""
    st.session_state.pop("analysis", None)


def run_analysis(resume_text: str, jd_text: str, use_llm: bool) -> dict:
    coverage = scorer.score_chunked(resume_text, jd_text)
    matched, missing, extra = compare_skills(resume_text, jd_text)
    result = {
        "coverage": coverage,
        "tfidf": score_tfidf(resume_text, jd_text),
        "whole_doc": scorer.score_whole_doc(resume_text, jd_text),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "quality": analyze_quality(resume_text),
        "grammar": check_grammar(resume_text, nlp),
        "llm": None,
        "llm_error": None,
    }
    if use_llm:
        try:
            result["llm"] = analyze_with_ollama(resume_text, jd_text, missing)
        except LLMUnavailable as e:
            result["llm_error"] = str(e)
    return result


def fmt_pct(x):
    return "n/a" if x is None else f"{x:.0%}"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("📄 ResumeRefiner AI")
st.caption(
    "Match a resume against a job description: semantic coverage, skill gaps, "
    "resume quality checks and optional local-LLM rewrite suggestions."
)

tab_analyze, tab_eval, tab_about = st.tabs(["Analyze", "Evaluate", "How it works"])


# ----------------------------------------------------------------------------
# Analyze
# ----------------------------------------------------------------------------

with tab_analyze:
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Try with sample data", on_click=load_sample, width="stretch")
    b2.button("Clear", on_click=clear_inputs, width="stretch")

    col_resume, col_jd = st.columns(2)

    with col_resume:
        st.subheader("1. Resume")
        uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        if uploaded is not None and uploaded.file_id != st.session_state.last_upload:
            try:
                st.session_state.resume_text = extract_text(uploaded, uploaded.name)
                st.session_state.last_upload = uploaded.file_id
                st.session_state.pop("analysis", None)
            except ExtractionError as e:
                st.error(str(e))
        st.text_area("...or paste / edit the resume text", key="resume_text", height=320)
        if st.session_state.resume_text:
            n_tok = scorer.count_tokens(st.session_state.resume_text)
            n_chunks = len(scorer.chunk(st.session_state.resume_text))
            st.caption(
                f"{len(st.session_state.resume_text.split())} words · {n_tok} word pieces · "
                f"read as {n_chunks} chunk(s) (model limit {MODEL_HARD_LIMIT} per chunk)"
            )

    with col_jd:
        st.subheader("2. Job description")
        st.text_area("Paste the full job description", key="jd_text", height=390)

    use_llm = st.checkbox(
        f"Add rewrite suggestions from a local LLM (Ollama · `{OLLAMA_MODEL}`)",
        help="Requires Ollama running locally. Nothing leaves your machine.",
    )
    ready = bool(st.session_state.resume_text.strip() and st.session_state.jd_text.strip())
    if st.button("Analyze", type="primary", disabled=not ready, width="stretch"):
        with st.spinner("Analyzing..."):
            st.session_state.analysis = run_analysis(
                st.session_state.resume_text, st.session_state.jd_text, use_llm)

    res = st.session_state.get("analysis")
    if res:
        cov = res["coverage"]
        q = res["quality"]
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("JD coverage", fmt_pct(cov.score),
                  help="Mean, over JD requirements, of the best-matching resume chunk "
                       "(cosine similarity). Use to compare candidates, not as a pass mark.")
        jd_skill_total = len(res["matched"]) + len(res["missing"])
        m2.metric("Skills matched",
                  f"{len(res['matched'])}/{jd_skill_total}" if jd_skill_total else "n/a")
        m3.metric("Keyword overlap (TF-IDF)", fmt_pct(res["tfidf"]))
        m4.metric("Resume quality", f"{q.score}/100")

        st.download_button(
            "Download report (Markdown)",
            build_markdown_report(cov, res["tfidf"], res["matched"], res["missing"],
                                  res["extra"], q, res["grammar"], res["llm"]),
            file_name="resume_report.md", mime="text/markdown",
        )

        t_match, t_quality, t_llm, t_debug = st.tabs(
            ["Match & skills", "Resume quality", "LLM suggestions", "Scorer details"])

        with t_match:
            s1, s2 = st.columns([1, 1])
            with s1:
                st.markdown("#### Skills")
                if res["missing"]:
                    st.error("**Required but missing:** " + ", ".join(res["missing"]))
                if res["matched"]:
                    st.success("**Matched:** " + ", ".join(res["matched"]))
                if res["extra"]:
                    st.info("**In resume, not requested:** " + ", ".join(res["extra"]))
                if not (res["missing"] or res["matched"]):
                    st.write("No known skills detected in the job description.")
            with s2:
                st.markdown("#### Weakest-covered requirements")
                st.caption("Where the resume says least about what the JD asks for.")
                if not cov.detail.empty:
                    st.dataframe(
                        cov.detail.head(8),
                        hide_index=True, width="stretch",
                        column_config={"Coverage": st.column_config.ProgressColumn(
                            "Coverage", min_value=0.0, max_value=1.0, format="%.2f")},
                    )

        with t_quality:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Words", q.word_count)
            k2.metric("Bullets", q.n_bullets)
            k3.metric("Start with action verb", fmt_pct(q.pct_action_verb))
            k4.metric("Contain a number", fmt_pct(q.pct_quantified))

            c1, c2 = st.columns(2)
            c1.markdown("**Sections:** " + " · ".join(
                f"{'✅' if ok else '❌'} {name}" for name, ok in q.sections.items()))
            c2.markdown("**Contact:** " + " · ".join(
                f"{'✅' if ok else '❌'} {name}" for name, ok in q.contact.items()))

            st.markdown("#### Suggestions")
            if not q.findings:
                st.success("No issues found by the rule-based checks.")
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
            for f in q.findings:
                with st.expander(f"{icon[f.severity]} {f.title}", expanded=f.severity == "high"):
                    st.write(f.detail)
                    for ex in f.examples:
                        st.markdown(f"> {ex}")

            if res["grammar"]:
                st.markdown("#### Grammar")
                for g in res["grammar"]:
                    st.write("- " + g)

        with t_llm:
            if res["llm"]:
                st.markdown(res["llm"])
            elif res["llm_error"]:
                st.warning(res["llm_error"])
            else:
                st.info("Tick the LLM option above and re-run to get bullet rewrites.")

        with t_debug:
            st.write(
                f"Resume split into **{cov.n_resume_chunks}** chunks, JD into "
                f"**{cov.n_jd_chunks}**."
            )
            st.dataframe(pd.DataFrame([
                {"Scorer": "Chunked embedding (v2)", "Score": cov.score},
                {"Scorer": "Whole-doc embedding (v1, truncated)", "Score": res["whole_doc"]},
                {"Scorer": "TF-IDF baseline", "Score": res["tfidf"]},
            ]), hide_index=True)
            st.caption("MiniLM cosine scores rarely approach 0, so read them relative "
                       "to each other rather than as absolute percentages.")


# ----------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------

with tab_eval:
    st.markdown(
        "Measure how well each scorer agrees with human judgement. Upload a CSV with "
        "columns **resume_text**, **jd_text**, **label**: an ordinal fit rating "
        "(1-5) or binary (0/1). Thirty to fifty hand-labelled pairs is enough to "
        "say something honest."
    )
    st.download_button("Download CSV template", template_csv(),
                       "eval_template.csv", "text/csv")

    eval_file = st.file_uploader("Labelled CSV", type=["csv"], key="eval_upload")
    if eval_file is not None:
        df = pd.read_csv(eval_file)
        if not REQUIRED_COLUMNS.issubset(df.columns):
            st.error(f"CSV must contain: {', '.join(sorted(REQUIRED_COLUMNS))}")
        else:
            df = df.dropna(subset=list(REQUIRED_COLUMNS))
            st.write(f"Loaded {len(df)} labelled pairs.")
            if st.button("Run evaluation", type="primary"):
                bar = st.progress(0.0, text="Scoring pairs...")
                summary, scored = evaluate(
                    df, scorer,
                    on_progress=lambda i, n: bar.progress(i / n, text=f"Scored {i}/{n}"))
                bar.empty()
                st.subheader("Agreement with human labels")
                st.dataframe(summary, hide_index=True, width="stretch")
                st.caption(
                    "Spearman is the headline metric: it measures whether a scorer "
                    "*ranks* candidates the way a human would. Empty cells mean "
                    "there wasn't enough label variety to compute that metric."
                )
                st.subheader("Raw scores")
                st.dataframe(scored, hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------

with tab_about:
    st.markdown(f"""
### Pipeline

1. **Extract** text from PDF (pdfplumber) or DOCX (mammoth).
2. **Chunk** both documents: split on line breaks, fall back to spaCy sentences for
   long lines, then greedily pack into {MAX_CHUNK_TOKENS}-token chunks.
3. **Embed** every chunk with `{SENTENCE_MODEL}` and compute **JD coverage**: for each
   JD chunk take the best-matching resume chunk, then average (max-then-mean).
4. **Match skills** with boundary-aware regex and an alias table.
5. **Check quality** with transparent rules: sections, contact info, action verbs,
   quantified bullets, weak phrases.
6. **Optionally** ask a local LLM (Ollama) for concrete bullet rewrites.

### Why chunking
`{SENTENCE_MODEL}` truncates at {MODEL_HARD_LIMIT} word pieces; a two-page resume is
~900. Encoding it whole means the vector describes the header and first job only.

### Why regex over substring matching
`"java" in "javascript"` is `True`. Neighbouring characters must be outside
`[A-Za-z0-9+#_]`; plain `\\b` fails because there's no word boundary after `C++`.

### Known limitations
- The skill list is curated, so unlisted skills are invisible.
- Quality checks are heuristics; grammar is a single rule.
- Cosine scores aren't calibrated probabilities. Rank with them; don't threshold.
- No bias audit: names, schools and gendered language can influence embeddings.

<sub>v{__version__}</sub>
""", unsafe_allow_html=True)
