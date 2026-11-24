"""Central configuration. Override the Ollama model with the OLLAMA_MODEL env var."""

import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
NLP_MODEL = "en_core_web_sm"
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates at 256 word pieces. We budget 200 to leave room
# for special tokens and to avoid landing exactly on the boundary.
MAX_CHUNK_TOKENS = 200
MODEL_HARD_LIMIT = 256

# Resume chunks are packed to a smaller budget than the hard cap: a ~3-bullet
# window keeps enough context to embed well while still pointing at a specific
# part of the resume when we show "best matching section".
RESUME_CHUNK_TOKENS = 80

# JD lines shorter than this are headings ("Requirements"), not requirements.
MIN_REQUIREMENT_WORDS = 4
