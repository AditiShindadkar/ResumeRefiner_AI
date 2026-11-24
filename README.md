# ResumeRefiner AI

[![tests](https://github.com/AditiShindadkar/ResumeRefiner_AI/actions/workflows/tests.yml/badge.svg)](https://github.com/AditiShindadkar/ResumeRefiner_AI/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Vercel](https://img.shields.io/badge/deploy-Vercel-000000)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

### 🔗 [**Try it live → resume-refiner-ai-lemon.vercel.app**](https://resume-refiner-ai-lemon.vercel.app)

Upload a resume and paste a job description. ResumeRefiner AI shows **which requirements the
resume covers, which skills are missing, and how to make the bullets stronger**, then lets you
download the result as a report. It runs on **Vercel** as a static page plus small Python
serverless functions; a Streamlit version is included for local use.

## Using the live app

1. Open the [live app](https://resume-refiner-ai-lemon.vercel.app) and click **Try with sample data**
   for a demo, or upload your own resume (PDF, DOCX or TXT) and paste a job description.
2. Optionally tick **Add rewrite suggestions from an LLM** for AI-written bullet rewrites.
3. Click **Analyze**. Results are split into four tabs: *Match & skills*, *Resume quality*,
   *LLM suggestions* and *Scorer details*. **Download report** saves everything as Markdown.

The first analysis takes a little longer while the browser downloads the ~23 MB embedding
model; after that it is cached. Works best in a recent Chrome, Edge, Firefox or Safari on desktop.

**Privacy:** nothing is stored. Resume text goes to the app's serverless functions for the
rule-based checks, and to [Groq](https://groq.com) only if you tick the LLM option. The semantic
matching runs entirely in your browser. The sample resume is fictional.

## Features

| | |
|---|---|
| **Semantic JD coverage** | Each job requirement is matched against every part of the resume with sentence embeddings, so the full resume is read, not just the first 256 tokens. |
| **Skill gap analysis** | ~70 skills with aliases (`sklearn` = `scikit-learn`) and boundary-aware matching (`JavaScript` doesn't count as `Java`, `HTML` doesn't count as `ML`). |
| **Resume quality checks** | Missing sections, missing contact info, bullets without action verbs, bullets without numbers, weak phrases ("responsible for", "worked on"), first-person pronouns, length. |
| **Weakest-covered requirements** | The JD lines your resume says least about, each paired with its closest resume section. |
| **LLM rewrite suggestions (optional)** | An LLM on [Groq](https://groq.com) (or a local [Ollama](https://ollama.com) model in the Streamlit app) quotes weak bullets and rewrites them. |
| **Downloadable report** | One-click Markdown report of all of the above. |
| **Evaluation harness** (Streamlit app) | Score a labelled CSV of (resume, JD, human rating) pairs and compare three scorers with Spearman / Pearson / ROC-AUC. |

## Deploy your own copy on Vercel

1. Fork this repo and import it at [vercel.com/new](https://vercel.com/new).
   No framework preset or build command is needed: `vercel.json` serves `public/` and turns
   each file in `api/` into a Python function.
2. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys) and add it
   under **Project → Settings → Environment Variables** as `GROQ_API_KEY`.
   Optionally set `GROQ_MODEL` (default `openai/gpt-oss-120b`).
3. Redeploy. Without a key, everything except the LLM tab still works.

Or from the CLI: `npm i -g vercel`, then `vercel`, then `vercel env add GROQ_API_KEY`, then `vercel --prod`.

> Anyone who can open the site can spend your Groq quota through the LLM tab. Groq's free
> tier is rate-limited, so the worst case is hitting that limit rather than a bill.

### What runs where

| Piece | Runs on | Why |
|---|---|---|
| UI | Static files in `public/` | No build step |
| PDF/DOCX extraction, skills, quality rules, TF-IDF | Python functions in `api/` | Reuses the tested `resume_refiner` package; only `pdfplumber` + `mammoth` are installed |
| Sentence embeddings (JD coverage) | **The visitor's browser**, via [transformers.js](https://github.com/huggingface/transformers.js) | PyTorch is far bigger than Vercel's 250 MB function limit. The ~23 MB quantized model downloads once and is cached by the browser |
| LLM feedback | Groq API, called from `api/suggest.py` | Keeps the API key on the server |

Scores from the in-browser (quantized) model are within about 0.01 of the Python
sentence-transformers scores on the sample data.

## Run locally

**Same as production** (standard library only, apart from `pip install -r requirements.txt`):

```bash
pip install -r requirements.txt
echo GROQ_API_KEY=gsk_your_key_here > .env      # optional; .env is git-ignored
python dev_server.py                             # http://localhost:8000
```

**Streamlit app** (full stack, including the evaluation harness and spaCy grammar check):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Click **Try with sample data** to see a full analysis without uploading anything. For LLM
feedback, set `GROQ_API_KEY`; otherwise the Streamlit app falls back to local
[Ollama](https://ollama.com) (`ollama pull mistral`, model set with `OLLAMA_MODEL`).

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

The Streamlit app's **Evaluate** tab takes a CSV with `resume_text`, `jd_text` and `label` columns (a 1–5 fit
rating or 0/1) and reports how well each scorer agrees with human judgement:

- **TF-IDF cosine**: lexical baseline
- **Whole-document embedding**: the naive (truncated) approach
- **Chunked coverage**: this project's method

Spearman ρ is the headline metric, because a screening tool's job is to rank candidates. A
template CSV can be downloaded from the tab.

## Project structure

```
public/                    Vercel front end (static)
  index.html, style.css
  app.js                   UI, API calls, in-browser embeddings
  chunking.js              JS port of chunking + coverage scoring
  report.js                JS port of the Markdown report
  samples/                 sample resume and job description (fictional)
api/                       Vercel Python functions: extract, analyze, suggest, health
dev_server.py              local stand-in for Vercel
app.py                     Streamlit UI (local)
resume_refiner/
  config.py                model names and token budgets
  skills.py                skill vocabulary and boundary-aware matching
  chunking.py              token-budgeted chunking and requirement splitting
  scoring.py               whole-doc and chunked-coverage embedding scorers
  tfidf.py                 dependency-free TF-IDF (matches scikit-learn)
  quality.py               rule-based resume quality checks
  extract.py               PDF / DOCX / TXT text extraction
  llm.py                   Groq and Ollama feedback
  web.py                   JSON endpoints used by api/ and dev_server.py
  evaluation.py            agreement metrics against human labels
  report.py                Markdown report builder
tests/                     unit tests (run without downloading models)
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake tokenizer and embedding model, so they run in a few seconds and need no
downloads. They also check that the pure-Python TF-IDF matches scikit-learn, and (when Node.js is
installed) that `public/chunking.js` produces the same chunks as the Python version. CI runs them
on Python 3.10–3.12.

## Limitations

- The skill vocabulary is curated, so skills outside it aren't detected.
- Quality checks are transparent heuristics, not a grammar engine. The spaCy grammar rule only
  runs in the Streamlit app.
- Cosine scores aren't calibrated probabilities. Use them to compare candidates, not as a pass mark.
- No bias audit: names, schools and gendered language can influence embeddings. Strip personal
  details before any real screening use.

## License

[MIT](LICENSE)
