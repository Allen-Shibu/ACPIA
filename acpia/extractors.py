"""Local text extraction from evidence files.

We deliberately extract text client-side rather than relying on Supermemory's
file-upload pipeline: the local build's PDF/webpage processing is unreliable, and
extracting ourselves means we control and can audit exactly what text enters the
case memory.
"""

import pathlib

TEXT_SUFFIXES = {".txt", ".md", ".log", ".csv", ".json", ".eml", ".vtt", ".srt"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


class UnsupportedFileType(Exception):
    pass


def extract_text(path: pathlib.Path) -> str:
    """Return the text content of an evidence file, or raise UnsupportedFileType."""
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(errors="replace")

    if suffix in PDF_SUFFIXES:
        return _extract_pdf(path)

    if suffix in IMAGE_SUFFIXES:
        raise UnsupportedFileType(
            f"'{path.name}' is an image. Image evidence (OCR / visual analysis) "
            "is not yet supported — extract any text manually for now."
        )

    raise UnsupportedFileType(
        f"'{path.name}' has unsupported type '{suffix}'. "
        f"Supported: {', '.join(sorted(TEXT_SUFFIXES | PDF_SUFFIXES))}."
    )


def _extract_pdf(path: pathlib.Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(f"[page {i}]\n{page_text}")
    text = "\n\n".join(parts)
    if not text.strip():
        raise UnsupportedFileType(
            f"'{path.name}' produced no extractable text — it may be a scanned "
            "image PDF requiring OCR, which is not yet supported."
        )
    return text
