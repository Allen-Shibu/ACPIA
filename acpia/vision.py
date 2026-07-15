"""Local vision analysis of image evidence — describe pixels + OCR visible text.

DOMAIN-CRITICAL: this is the ONE place ACPIA reads image pixels rather than just
EXIF metadata. In a child-protection context, suspect imagery MUST NOT leave the
machine, so this refuses to run against a non-local LLM endpoint. The guardrail is
code, not a promise: if OPENAI_BASE_URL is not localhost, we raise instead of
sending pixels anywhere.

Output is descriptive lead material, flagged unverified — a neutral account of
what is visibly present, never an identity/age/intent determination.
"""

import base64
import os
import pathlib
from urllib.parse import urlparse

from openai import OpenAI

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

_PROMPT = (
    "You are describing a piece of image evidence for an investigator's records. "
    "State ONLY what is visibly present, factually and neutrally: objects, setting, "
    "and the number/clothing of any people. Transcribe any visible text verbatim. "
    "Do NOT guess identities, ages, relationships, or intent, and do NOT assess risk "
    "— that is for a human. If something is unclear, say it is unclear."
)


class CloudVisionRefused(RuntimeError):
    """Raised when vision is attempted against a non-local endpoint."""


def _is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in _LOCAL_HOSTS


def describe_image(path: pathlib.Path) -> str:
    """Return a neutral visual description + OCR of an image, using a LOCAL vision model.

    Raises CloudVisionRefused if OPENAI_BASE_URL is not a local endpoint — pixels
    must never be sent to a cloud provider in this domain.
    """
    base_url = os.environ["OPENAI_BASE_URL"]
    if not _is_local(base_url):
        raise CloudVisionRefused(
            f"Refusing to send image pixels to non-local endpoint '{base_url}'. "
            "Point OPENAI_BASE_URL at a local model (e.g. Ollama) to use --vision."
        )

    mime = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else path.suffix.lower().lstrip(".")
    b64 = base64.b64encode(path.read_bytes()).decode()
    llm = OpenAI(base_url=base_url, api_key=os.environ["OPENAI_API_KEY"])
    resp = llm.chat.completions.create(
        model=os.environ.get("OPENAI_VISION_MODEL", "llama3.2-vision"),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


def _demo() -> None:
    """Self-check the guardrail without any network call."""
    assert _is_local("http://localhost:11434/v1")
    assert _is_local("http://127.0.0.1:11434/v1")
    assert not _is_local("https://openrouter.ai/api/v1")
    assert not _is_local("https://generativelanguage.googleapis.com/v1beta/openai/")
    print("vision guardrail self-check OK")


if __name__ == "__main__":
    _demo()
