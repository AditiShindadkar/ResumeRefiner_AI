"""Optional recruiter-style feedback from a local LLM via Ollama."""

from typing import List, Optional

from .config import OLLAMA_MODEL

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


class LLMUnavailable(Exception):
    pass


def analyze_with_ollama(resume_text: str, jd_text: str, missing_skills: List[str],
                        model: str = OLLAMA_MODEL) -> Optional[str]:
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable("The `ollama` Python package is not installed.") from e

    prompt = PROMPT.format(
        missing=", ".join(missing_skills) if missing_skills else "none",
        resume=resume_text,
        jd=jd_text,
    )
    try:
        r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        raise LLMUnavailable(
            f"Could not reach Ollama ({e}). Start it with `ollama serve` and "
            f"pull the model with `ollama pull {model}`."
        ) from e
    return r["message"]["content"]
