"""Plain-text extraction from PDF, DOCX and TXT uploads."""

from typing import Optional


class ExtractionError(Exception):
    """Raised with a user-facing message when a file can't be read."""


def extract_text_from_pdf(file) -> Optional[str]:
    try:
        import pdfplumber
    except ImportError as e:
        raise ExtractionError("pdfplumber is not installed: `pip install pdfplumber`") from e
    try:
        with pdfplumber.open(file) as pdf:
            pages = [p.extract_text() for p in pdf.pages]
    except Exception as e:
        raise ExtractionError(f"Could not read PDF: {e}") from e
    text = "\n".join(p for p in pages if p)
    if not text.strip():
        raise ExtractionError(
            "No text found in this PDF. It may be a scanned image; "
            "export it from Word/Docs as a text PDF or upload the DOCX instead."
        )
    return text


def extract_text_from_docx(file) -> Optional[str]:
    try:
        import mammoth
    except ImportError as e:
        raise ExtractionError("mammoth is not installed: `pip install mammoth`") from e
    try:
        text = mammoth.extract_raw_text(file).value.strip()
    except Exception as e:
        raise ExtractionError(f"Could not read DOCX: {e}") from e
    if not text:
        raise ExtractionError("This DOCX file appears to be empty.")
    return text


def extract_text(file, filename: str) -> str:
    """Dispatch on file extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    if name.endswith(".docx"):
        return extract_text_from_docx(file)
    if name.endswith(".txt"):
        data = file.read()
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    raise ExtractionError(f"Unsupported file type: {filename}")
