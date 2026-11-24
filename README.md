# ResumeRefiner AI

[![tests](https://github.com/AditiShindadkar/ResumeRefiner_AI/actions/workflows/tests.yml/badge.svg)](https://github.com/AditiShindadkar/ResumeRefiner_AI/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Upload a resume and paste a job description. ResumeRefiner AI shows **which requirements the
resume covers, which skills are missing, and how to make the bullets stronger**, then lets you
download the result as a report. Everything runs locally: no resume text is sent to a
third-party API.

## Features

| | |
|---|---|
| **Semantic JD coverage** | Each job requirement is matched against every part of the resume with sentence embeddings, so the full resume is read, not just the first 256 tokens. |
| **Skill gap analysis** | ~70 skills with aliases (`sklearn` = `scikit-learn`) and boundary-aware matching (`JavaScript` doesn't count as `Java`, `HTML` doesn't count as `ML`). |
| **Resume quality checks** | Missing sections, missing contact info, bullets without action verbs, bullets without numbers, weak phrases ("responsible for", "worked on"), first-person pronouns, length. |
| **Weakest-covered requirements** | The JD lines your resume says least about, each paired with its closest resume section. |
| **LLM rewrite suggestions (optional)** | A local model via [Ollama](https://ollama.com) quotes weak bullets and rewrites them. |
| **Downloadable report** | One-click Markdown report of all of the above. |
| **Evaluation harness** | Score a labelled CSV of (resume, JD, human rating) pairs and compare three scorers with Spearman / Pearson / ROC-AUC. |

## Quick start

```bash
git clone https://github.com/AditiShindadkar/ResumeRefiner_AI.git
cd ResumeRefiner_AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Click **Try with sample data** to see a full analysis without uploading anything.

**Optional LLM feedback:** install [Ollama](https://ollama.com), run `ollama pull mistral`,
then tick the LLM option in the app. Use a different model with `OLLAMA_MODEL=llama3 streamlit run app.py`.

## How it works

```
 PDF / DOCX / TXT
        │  pdfplumber / mammoth
        ▼
   plain text ──► skill matcher (regex + alias table) ──► matched / missing / extra
        │
        ├──► quality rules ──► sections, verbs, numbers, weak phrases ──► score /100
        │
        ▼
 chunking ── resume: neighbouring lines packed into ≤80-token chunks
        │    JD:     one chunk per requirement line (headings dropped)
        ▼
 all-MiniLM-L6-v2 embeddings
        │
        ▼
 JD coverage = mean over requirements of max cosine similarity to any resume chunk
```

**Why chunk?** `all-MiniLM-L6-v2` silently truncates input at 256 word pieces, and a two-page
resume is about 900. Encoding the whole resume as one string means the vector only describes the
header and first job.

**Why max-then-mean?** For each requirement we ask "is this covered *anywhere*?" Averaging over
all resume chunks instead would penalise a candidate for relevant experience the JD didn't
mention.

**Why regex instead of `in`?** `"java" in "javascript"` is `True`. Neighbouring characters must
fall outside `[A-Za-z0-9+#_]`; plain `\b` fails because there's no word boundary after the `+`
in `C++`.

## Evaluating the scorer

The **Evaluate** tab takes a CSV with `resume_text`, `jd_text` and `label` columns (a 1–5 fit
rating or 0/1) and reports how well each scorer agrees with human judgement:

- **TF-IDF cosine**: lexical baseline
- **Whole-document embedding**: the naive (truncated) approach
- **Chunked coverage**: this project's method

Spearman ρ is the headline metric, because a screening tool's job is to rank candidates. A
template CSV can be downloaded from the tab.

## Project structure

```
app.py                     Streamlit UI
resume_refiner/
  config.py                model names and token budgets
  skills.py                skill vocabulary and boundary-aware matching
  chunking.py              token-budgeted chunking and requirement splitting
  scoring.py               TF-IDF, whole-doc and chunked-coverage scorers
  quality.py               rule-based resume quality checks
  extract.py               PDF / DOCX / TXT text extraction
  llm.py                   optional Ollama feedback
  evaluation.py            agreement metrics against human labels
  report.py                Markdown report builder
data/                      sample resume and job description (fictional)
tests/                     unit tests (run without downloading models)
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake tokenizer and embedding model, so they run in a few seconds and need no
downloads. CI runs them on Python 3.10–3.12.

## Limitations

- The skill vocabulary is curated, so skills outside it aren't detected.
- Quality checks are transparent heuristics, not a grammar engine.
- Cosine scores aren't calibrated probabilities. Use them to compare candidates, not as a pass mark.
- No bias audit: names, schools and gendered language can influence embeddings. Strip personal
  details before any real screening use.

## License

[MIT](LICENSE)
