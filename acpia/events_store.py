"""Local per-case store of structured, ground-truth events.

Holds events extracted directly from source metadata (chat timestamps, image EXIF)
— the reliable, non-AI half of the timeline. Stored as a JSON file per case under
.acpia/ in the project directory.
"""

import json
import pathlib
import uuid

_ROOT = pathlib.Path(".acpia")


def _case_path(case: str) -> pathlib.Path:
    return _ROOT / f"{case}.events.json"


def list_cases() -> list[str]:
    """All case IDs known locally (from .acpia/*.events.json stems).
    ponytail: text-only cases with no chat/image metadata leave no event file, so
    they won't appear here — swap for a real case registry if that becomes common."""
    if not _ROOT.exists():
        return []
    return sorted(p.name[: -len(".events.json")] for p in _ROOT.glob("*.events.json"))


def load_events(case: str) -> list[dict]:
    path = _case_path(case)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def add_events(case: str, events: list[dict]) -> int:
    """Append structured events to a case store. Returns count added."""
    _ROOT.mkdir(exist_ok=True)
    existing = load_events(case)
    for e in events:
        e.setdefault("id", uuid.uuid4().hex[:12])
    existing.extend(events)
    _case_path(case).write_text(json.dumps(existing, indent=2))
    return len(events)


def make_chat_event(case: str, source_file: str, parsed: dict) -> dict:
    return {
        "case": case,
        "kind": "chat_message",
        "source_file": source_file,
        "timestamp": parsed["timestamp"],
        "timestamp_raw": parsed["timestamp_raw"],
        "actor": parsed["sender"],
        "content": parsed["message"],
        "extra": {},
    }


def make_image_event(case: str, source_file: str, meta: dict) -> dict:
    return {
        "case": case,
        "kind": "image_capture",
        "source_file": source_file,
        "timestamp": meta["timestamp"],
        "timestamp_raw": meta["timestamp_raw"],
        "actor": None,
        "content": f"Image captured{' by ' + meta['camera'] if meta.get('camera') else ''}",
        "extra": {"gps": meta.get("gps"), "camera": meta.get("camera")},
    }
