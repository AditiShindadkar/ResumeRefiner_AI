"""
Local stand-in for Vercel: serves public/ and routes /api/<name> to the same
functions the serverless handlers use. Standard library only.

    python dev_server.py            # http://localhost:8000
    GROQ_API_KEY=... python dev_server.py
"""

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resume_refiner.web import dispatch

ROOT = Path(__file__).parent
PUBLIC = ROOT / "public"


def load_dotenv(path: Path = ROOT / ".env"):
    """Read KEY=value lines from .env (git-ignored) without overriding the
    real environment. Lets Windows users skip shell-specific syntax."""
    if not path.exists():
        return
    raw = path.read_bytes()
    # PowerShell 5.1's `echo ... > .env` writes UTF-16 with a BOM.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class DevHandler(SimpleHTTPRequestHandler):
    def _api(self, body: bytes):
        status, data = dispatch(self.path.split("?")[0][len("/api/"):], body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self.send_error(405)
        self._api(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._api(b"{}")
        super().do_GET()


if __name__ == "__main__":
    load_dotenv()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("PORT", 8000))
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(DevHandler, directory=str(PUBLIC)))
    print(f"ResumeRefiner AI on http://localhost:{port}  (Ctrl+C to stop)")
    if os.getenv("GROQ_API_KEY"):
        print("Groq key: set")
    else:
        print("Groq key: not set, so the LLM tab is disabled.\n"
              "  Add a line  GROQ_API_KEY=gsk_...  to a .env file in this folder,\n"
              "  or in PowerShell run  $env:GROQ_API_KEY=\"gsk_...\"  before starting.")
    server.serve_forever()
