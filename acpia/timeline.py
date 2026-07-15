"""Assemble a case timeline by merging ground-truth structured events (real
metadata) with Supermemory prose memories (explicit or inferred dates)."""

from acpia import events_store
from acpia.dates import resolve_dates
from acpia.supermemory_client import SupermemoryClient


def build_timeline(case: str, use_llm: bool = True) -> tuple[list[dict], list[dict]]:
    """Return (dated_entries_sorted, undated_entries).

    Each entry: {date, date_source, kind, text, source, citation}.
    date_source: 'metadata' (real), 'explicit' (Supermemory), 'inferred' (LLM).
    """
    entries = []

    # 1. Ground-truth structured events (chat timestamps, image EXIF).
    for e in events_store.load_events(case):
        if e["kind"] == "chat_message":
            text = f"{e['actor']}: {e['content']}"
        else:
            text = e["content"]
            if e.get("extra", {}).get("gps"):
                g = e["extra"]["gps"]
                text += f" (GPS {g['lat']:.4f},{g['lon']:.4f})"
        entries.append(
            {
                "date": e["timestamp"],
                "date_raw": e.get("timestamp_raw"),
                "date_source": "metadata",
                "kind": e["kind"],
                "text": text,
                "source": e["source_file"],
                "citation": None,
            }
        )

    # 2. Supermemory prose memories, dated in trust order.
    client = SupermemoryClient()
    memories = client.fetch_all_memories(case)
    for m in resolve_dates(memories, use_llm=use_llm):
        entries.append(
            {
                "date": m["date"],
                "date_raw": None,
                "date_source": m["date_source"],
                "kind": "memory",
                "text": m["memory"],
                "source": m.get("source_title"),
                "citation": m["id"],
            }
        )

    dated = [e for e in entries if e["date"]]
    undated = [e for e in entries if not e["date"]]
    dated.sort(key=lambda e: e["date"])
    return dated, undated
