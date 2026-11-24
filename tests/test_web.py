import base64
import io
import json

import pytest

from resume_refiner import llm
from resume_refiner.web import MAX_TEXT_CHARS, dispatch


def call(name, payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    status, data = dispatch(name, body)
    return status, json.loads(data)


def test_analyze_returns_skills_quality_and_tfidf():
    status, out = call("analyze", {
        "resume": "EXPERIENCE\n- Built Python reports for 3 teams\nSKILLS\nPython, Tableau",
        "jd": "We need Python and Snowflake experience",
    })
    assert status == 200
    assert out["skills"] == {"matched": ["Python"], "missing": ["Snowflake"], "extra": ["Tableau"]}
    assert 0 <= out["quality"]["score"] <= 100
    assert {"severity", "title", "detail", "examples"} <= set(out["quality"]["findings"][0])
    assert 0 < out["tfidf"] <= 1
    assert out["chunking"]["resume_chunk_tokens"] <= out["chunking"]["max_chunk_tokens"]


@pytest.mark.parametrize("payload, status", [
    ({"resume": "", "jd": "x"}, 400),
    ({"resume": 5, "jd": "x"}, 400),
    ({"resume": "x" * (MAX_TEXT_CHARS + 1), "jd": "x"}, 413),
    (b"not json", 400),
    (b"[1, 2]", 400),
])
def test_analyze_rejects_bad_input(payload, status):
    code, out = call("analyze", payload)
    assert code == status and out["error"]


def test_unknown_endpoint():
    assert call("nope", {})[0] == 404


def test_extract_txt_and_errors():
    data = base64.b64encode("Hello resume".encode()).decode()
    assert call("extract", {"filename": "cv.txt", "data": data}) == (200, {"text": "Hello resume"})
    assert call("extract", {"filename": "cv.exe", "data": data})[0] == 422
    assert call("extract", {"filename": "cv.txt", "data": "%%%"})[0] == 400
    assert call("extract", {"filename": "cv.txt", "data": ""})[0] == 400


def test_suggest_without_key_is_503(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code, out = call("suggest", {"resume": "r", "jd": "j", "missing": []})
    assert code == 503 and "GROQ_API_KEY" in out["error"]


def test_suggest_calls_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout):
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return FakeResponse(json.dumps(
            {"choices": [{"message": {"content": "## Overall fit\nGood."}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    code, out = call("suggest", {"resume": "my resume", "jd": "the jd", "missing": ["SQL"]})
    assert code == 200 and out["feedback"].startswith("## Overall fit")
    assert seen["auth"] == "Bearer test-key"
    prompt = seen["body"]["messages"][0]["content"]
    assert "my resume" in prompt and "SQL" in prompt


def test_health(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert call("health", {})[1]["llm_configured"] is False
