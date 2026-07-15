"""Resolve event dates for Supermemory prose memories, in trust order.

Tier 1: Supermemory's structured `temporalContext.eventDate` (explicit) — trusted.
Tier 2: LLM inference from the memory text, anchored to the source document date —
        flagged `inferred`, because LLMs miscalculate relative dates (e.g. weekday math).
Unresolved memories are returned with date=None (the "undated" bucket).
"""

import json
import os
import re

from openai import OpenAI

# An explicit, unambiguous date written verbatim in the text (not a relative phrase).
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _explicit_date(memory: dict) -> str | None:
    tc = (memory.get("metadata") or {}).get("temporalContext") or {}
    ev = tc.get("eventDate")
    if isinstance(ev, list) and ev:
        return ev[0]
    if isinstance(ev, str):
        return ev
    return None


def _verbatim_date(text: str) -> str | None:
    """A date literally written in the memory text (e.g. '2026-06-03'). Trusted:
    the LLM/extractor read it, it wasn't calculated from a relative reference."""
    m = _ISO_RE.search(text or "")
    return "-".join(m.groups()) if m else None


def resolve_dates(memories: list[dict], use_llm: bool = True) -> list[dict]:
    """Return memories annotated with {date, date_source} where date_source is
    'explicit', 'inferred', or None."""
    out = []
    needs_llm = []
    for m in memories:
        d = _explicit_date(m) or _verbatim_date(m.get("memory", ""))
        if d:
            out.append({**m, "date": d, "date_source": "explicit"})
        else:
            entry = {**m, "date": None, "date_source": None}
            out.append(entry)
            needs_llm.append(entry)

    if use_llm and needs_llm:
        _infer_dates_llm(needs_llm)
    return out


def _infer_dates_llm(entries: list[dict]) -> None:
    """Mutate entries in place, filling date/date_source via a single batched call."""
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]
    )
    lines = []
    for i, e in enumerate(entries):
        anchor = e.get("source_created_at") or "unknown"
        lines.append(f"{i}. (recorded {anchor}) {e['memory']}")
    prompt = (
        "For each numbered statement, extract the date of the EVENT it describes as "
        "ISO 8601 (YYYY-MM-DD, add THH:MM if a time is given). Resolve relative terms "
        "(yesterday, last Tuesday, last year) against that item's 'recorded' anchor date. "
        "Assume events are in the past relative to the anchor. If no date is determinable, "
        "use null. Only use information present in the text; never invent a date.\n"
        'Return JSON: {"dates": [{"n": <int>, "iso": <string|null>}]}.\n\n'
        + "\n".join(lines)
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        by_n = {d["n"]: d.get("iso") for d in data.get("dates", [])}
        for i, e in enumerate(entries):
            iso = by_n.get(i)
            if iso:
                e["date"] = iso
                e["date_source"] = "inferred"
    except Exception:
        # If inference fails, entries simply stay undated — never fabricate.
        pass
