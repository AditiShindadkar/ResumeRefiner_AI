"""Vercel serverless function: /api/health (logic in resume_refiner/web.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_refiner.web import EndpointHandler  # noqa: E402


class handler(EndpointHandler):
    endpoint = "health"
