"""Parse chat/message exports into structured, real-timestamped events.

Unlike prose facts extracted by Supermemory, these carry the *actual* per-message
timestamp from the source file — ground truth, not AI inference.
"""

import re
from datetime import datetime

_LINE_RE = re.compile(
    r"^\s*[\[\(]?\s*"
    r"(?P<ts>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T,]+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)"
    r"\s*[\]\)]?\s*[-–]?\s*"
    r"(?P<sender>[^:]{1,40}?)\s*:\s*"
    r"(?P<msg>.+)$",
    re.IGNORECASE,
)

# Explicit formats (auditable) tried in order against the captured timestamp string.
_TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m/%d/%Y, %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y, %H:%M",
    "%d/%m/%y, %H:%M",
    "%m/%d/%y, %H:%M",
]


def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip().replace("T", " ")
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def looks_like_chat(text: str, min_matches: int = 2) -> bool:
    """Heuristic: does this text contain several timestamped message lines?"""
    matches = sum(1 for line in text.splitlines() if _LINE_RE.match(line))
    return matches >= min_matches


def parse_chat(text: str) -> list[dict]:
    """Return a list of {timestamp, timestamp_raw, sender, message} events.

    timestamp is an ISO string when parseable, else None (raw kept regardless).
    """
    events = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        dt = _parse_ts(m["ts"])
        events.append(
            {
                "timestamp": dt.isoformat() if dt else None,
                "timestamp_raw": m["ts"].strip(),
                "sender": m["sender"].strip(),
                "message": m["msg"].strip(),
            }
        )
    return events
