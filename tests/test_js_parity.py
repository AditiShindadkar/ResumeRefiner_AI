"""The browser re-implements chunking in public/chunking.js; check it agrees
with the Python version. Skipped when Node.js isn't installed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from resume_refiner.chunking import chunk_text, split_requirements

ROOT = Path(__file__).parent.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

RESUME = (ROOT / "public" / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
JD = (ROOT / "public" / "samples" / "sample_jd.txt").read_text(encoding="utf-8")
LONG = " ".join(f"w{i}" for i in range(55))


def word_count(text):
    return len(text.split())


def run_js(cases):
    script = rf"""
    import * as c from {json.dumps((ROOT / "public" / "chunking.js").as_uri())};
    const wc = (t) => t.split(/\s+/).filter(Boolean).length;
    const cases = {json.dumps(cases)};
    const out = cases.map(([fn, text, max]) => fn === "chunk"
        ? c.chunkText(text, wc, max) : c.splitRequirements(text, wc, max, null, 4));
    console.log(JSON.stringify(out));
    """
    r = subprocess.run([NODE, "--input-type=module", "-e", script],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(r.stdout)


def test_chunking_matches_python():
    cases = [["chunk", RESUME, 80], ["chunk", RESUME, 20], ["chunk", LONG, 10],
             ["chunk", "", 10], ["req", JD, 200], ["req", JD, 6]]
    expected = [
        chunk_text(t, word_count, m) if fn == "chunk"
        else split_requirements(t, word_count, m, None, 4)
        for fn, t, m in cases
    ]
    assert run_js(cases) == expected
