"""
Recruiter-style feedback from an LLM.

Two backends share one prompt:
  * Groq (hosted, OpenAI-compatible API) - used by the Vercel deployment and
    by the Streamlit app when GROQ_API_KEY is set.
  * Ollama (local) - the original offline option for the Streamlit app.
"""

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .config import GROQ_MODEL, OLLAMA_MODEL

PROMPT = """You are an experienced technical recruiter.
Review this resume against the job description and give specific, actionable feedback.

Structure your answer with these headings:
1. Overall fit (2-3 sentences)
2. Keyword alignment
3. Bullet rewrites: quote 3 weak bullets verbatim and show an improved, quantified rewrite of each
4. What to add or remove

Do not invent experience the candidate doesn't have; where a metric is unknown,
use a placeholder like [X%].

Automated analysis found these required skills absent from the resume:
{missing}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMUnavailable(Exception):
    pass


def build_prompt(resume_text: str, jd_text: str, missing_skills: List[str]) -> str:
    return PROMPT.format(
        missing=", ".join(missing_skills) if missing_skills else "none",
        resume=resume_text,
        jd=jd_text,
    )


def analyze_with_groq(resume_text: str, jd_text: str, missing_skills: List[str],
                      model: Optional[str] = None, api_key: Optional[str] = None,
                      timeout: float = 45) -> str:
    """Call Groq's chat-completions endpoint with only the standard library,
    so the serverless bundle needs no SDK."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMUnavailable(
            "GROQ_API_KEY is not set. Create a free key at console.groq.com and add "
            "it as an environment variable (Vercel: Project → Settings → "
            "Environment Variables).")
    model = model or GROQ_MODEL
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": build_prompt(resume_text, jd_text, missing_skills)}],
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Groq sits behind Cloudflare, which rejects urllib's default User-Agent.
        "User-Agent": "ResumeRefinerAI/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e)["error"]["message"]
        except Exception:
            msg = e.reason
        if e.code == 429:
            msg = "Groq rate limit reached; wait a minute and try again. " + str(msg)
        raise LLMUnavailable(f"Groq API error ({e.code}, model `{model}`): {msg}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMUnavailable(f"Could not reach Groq: {e}") from e
    return data["choices"][0]["message"]["content"]


def analyze_with_ollama(resume_text: str, jd_text: str, missing_skills: List[str],
                        model: str = OLLAMA_MODEL) -> Optional[str]:
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable(
            "The `ollama` Python package is not installed (`pip install ollama`). "
            "Alternatively set GROQ_API_KEY to use Groq instead.") from e
    prompt = build_prompt(resume_text, jd_text, missing_skills)
    try:
        r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        raise LLMUnavailable(
            f"Could not reach Ollama ({e}). Start it with `ollama serve` and "
            f"pull the model with `ollama pull {model}`."
        ) from e
    return r["message"]["content"]


def analyze_with_llm(resume_text: str, jd_text: str, missing_skills: List[str]) -> str:
    """Groq when a key is configured, otherwise local Ollama."""
    if os.getenv("GROQ_API_KEY"):
        return analyze_with_groq(resume_text, jd_text, missing_skills)
    return analyze_with_ollama(resume_text, jd_text, missing_skills)


def llm_label() -> str:
    if os.getenv("GROQ_API_KEY"):
        return f"Groq · `{GROQ_MODEL}`"
    return f"local Ollama · `{OLLAMA_MODEL}`"
