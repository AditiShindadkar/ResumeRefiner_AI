"""
JSON endpoints for the Vercel deployment (see api/*.py).

Each endpoint is a plain function `dict -> dict` so it can be unit-tested and
served by both Vercel's Python runtime and dev_server.py. Only lightweight
modules are imported here: no pandas, scikit-learn, torch or spaCy.
"""

import base64
import binascii
import io
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler

from .config import MAX_CHUNK_TOKENS, MIN_REQUIREMENT_WORDS, RESUME_CHUNK_TOKENS

MAX_TEXT_CHARS = 30_000          # ~5 pages; keeps LLM token spend bounded
MAX_UPLOAD_BYTES = 3_000_000     # Vercel caps request bodies at 4.5 MB (base64 adds 33%)


class WebError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _text(payload: dict, key: str, required: bool = True) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise WebError(400, f"`{key}` must be a string.")
    value = value.strip()
    if required and not value:
        raise WebError(400, f"`{key}` is empty.")
    if len(value) > MAX_TEXT_CHARS:
        raise WebError(413, f"`{key}` is too long ({len(value):,} characters; "
                            f"limit {MAX_TEXT_CHARS:,}).")
    return value


def extract(payload: dict) -> dict:
    """{filename, data: base64} -> {text}"""
    from .extract import ExtractionError, extract_text

    filename = payload.get("filename") or ""
    try:
        raw = base64.b64decode(payload.get("data") or "", validate=True)
    except (binascii.Error, ValueError):
        raise WebError(400, "File data is not valid base64.")
    if not raw:
        raise WebError(400, "The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise WebError(413, "File is larger than 3 MB.")
    try:
        text = extract_text(io.BytesIO(raw), filename)
    except ExtractionError as e:
        raise WebError(422, str(e))
    return {"text": text[:MAX_TEXT_CHARS]}


def analyze(payload: dict) -> dict:
    """{resume, jd} -> skills, quality report, TF-IDF score, chunking config."""
    from .quality import analyze_quality
    from .skills import compare_skills
    from .tfidf import score_tfidf

    resume = _text(payload, "resume")
    jd = _text(payload, "jd")
    matched, missing, extra = compare_skills(resume, jd)
    return {
        "skills": {"matched": matched, "missing": missing, "extra": extra},
        "quality": asdict(analyze_quality(resume)),
        "tfidf": score_tfidf(resume, jd),
        # The browser does the embedding step; send it the same budgets.
        "chunking": {
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "resume_chunk_tokens": min(RESUME_CHUNK_TOKENS, MAX_CHUNK_TOKENS),
            "min_requirement_words": MIN_REQUIREMENT_WORDS,
        },
    }


def suggest(payload: dict) -> dict:
    """{resume, jd, missing: [..]} -> {feedback, model}"""
    from .config import GROQ_MODEL
    from .llm import LLMUnavailable, analyze_with_groq

    resume = _text(payload, "resume")
    jd = _text(payload, "jd")
    missing = payload.get("missing") or []
    if not isinstance(missing, list) or not all(isinstance(m, str) for m in missing):
        raise WebError(400, "`missing` must be a list of strings.")
    try:
        feedback = analyze_with_groq(resume, jd, missing[:50])
    except LLMUnavailable as e:
        raise WebError(503, str(e))
    return {"feedback": feedback, "model": GROQ_MODEL}


def health(_payload: dict) -> dict:
    import os

    from .config import GROQ_MODEL
    return {"ok": True, "llm_configured": bool(os.getenv("GROQ_API_KEY")),
            "llm_model": GROQ_MODEL}


ENDPOINTS = {"extract": extract, "analyze": analyze, "suggest": suggest, "health": health}


def dispatch(name: str, body: bytes):
    """Run an endpoint on a raw request body. Returns (status, json_bytes)."""
    func = ENDPOINTS.get(name)
    if func is None:
        status, out = 404, {"error": f"Unknown endpoint: {name}"}
    else:
        try:
            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise WebError(400, "Request body must be a JSON object.")
            status, out = 200, func(payload)
        except json.JSONDecodeError:
            status, out = 400, {"error": "Request body is not valid JSON."}
        except WebError as e:
            status, out = e.status, {"error": str(e)}
        except Exception as e:  # never leak a traceback to the browser
            status, out = 500, {"error": f"Internal error: {type(e).__name__}"}
    return status, json.dumps(out).encode()


class EndpointHandler(BaseHTTPRequestHandler):
    """Base for Vercel's `handler` classes. Subclasses set `endpoint`."""

    endpoint = ""

    def _send(self, status: int, data: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self._send(*dispatch(self.endpoint, self.rfile.read(length)))

    def do_GET(self):
        if self.endpoint == "health":
            self._send(*dispatch("health", b"{}"))
        else:
            self._send(405, b'{"error": "Use POST."}')
