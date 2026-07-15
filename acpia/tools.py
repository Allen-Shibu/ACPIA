import json

from acpia import events_store
from acpia.supermemory_client import SupermemoryClient

SEARCH_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": (
            "Semantic search over the evidence memories ingested for this case. "
            "Returns matching facts extracted from evidence, each with a memory ID "
            "you must cite when using it in your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'who did the suspect contact'",
                }
            },
            "required": ["query"],
        },
    },
}

BUILD_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "build_case_profile",
        "description": (
            "Compact this whole case into a structured, cited profile: people (with "
            "aliases/roles), identifiers (phones/emails/usernames/devices/vehicles), "
            "locations, and behaviors — each pointing to the memory IDs it came from. "
            "Use to orient on who and what is in the case."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

BUILD_TIMELINE_TOOL = {
    "type": "function",
    "function": {
        "name": "build_case_timeline",
        "description": (
            "Reconstruct a chronological timeline for this case, merging ground-truth "
            "metadata (chat/EXIF timestamps) with dated evidence. Use to understand "
            "the sequence of events. Dates marked (verify) are LLM-inferred."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

LIST_CASES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_other_cases",
        "description": (
            "List the IDs of OTHER cases known locally (excluding the current case). "
            "Use before correlating to see what this case could be linked against."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

CORRELATE_TOOL = {
    "type": "function",
    "function": {
        "name": "correlate_cases",
        "description": (
            "Find people, identifiers, and locations SHARED between this case and one "
            "or more other cases. Returns exact matches (hard links) and LLM soft "
            "matches (must be verified), each with cited memory IDs from every case. "
            "Use to surface cross-case leads."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "other_case_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Other case IDs to compare this case against (from list_other_cases).",
                }
            },
            "required": ["other_case_ids"],
        },
    },
}

TOOLS = [
    SEARCH_EVIDENCE_TOOL,
    BUILD_PROFILE_TOOL,
    BUILD_TIMELINE_TOOL,
    LIST_CASES_TOOL,
    CORRELATE_TOOL,
]


def _fmt_timeline(dated: list, undated: list) -> str:
    if not dated and not undated:
        return "No timeline events found for this case."
    lines = []
    for e in dated:
        flag = " (verify)" if e["date_source"] == "inferred" else ""
        cite = f" [memory:{e['citation']}]" if e["citation"] else ""
        lines.append(f"{e['date'].replace('T', ' ')}{flag}: {' '.join(e['text'].split())}{cite}")
    for e in undated:
        cite = f" [memory:{e['citation']}]" if e["citation"] else ""
        lines.append(f"(undated): {e['text']}{cite}")
    return "\n".join(lines)


def _fmt_correlate(result: dict) -> str:
    if not result["exact"] and not result["fuzzy"]:
        return "No shared entities found across these cases."
    lines = []
    for tier, label in (("exact", "EXACT LINK"), ("fuzzy", "SOFT LINK (verify)")):
        for link in result[tier]:
            lines.append(f"{label}: {link['value']}")
            for m in link["members"]:
                cites = " ".join(f"[memory:{c}]" for c in m["cites"])
                lines.append(f"  - case {m['case']}: {m['display']} {cites}")
    return "\n".join(lines)


def dispatch_tool_call(name: str, arguments: dict, client: SupermemoryClient, case_id: str) -> str:
    if name == "search_evidence":
        results = client.search(query=arguments["query"], container_tag=case_id)
        if not results:
            return "No matching evidence found."
        return "\n".join(
            f"[memory:{r['id']}] (similarity={r['similarity']:.2f}) {r['memory']}" for r in results
        )

    if name == "build_case_profile":
        from acpia.profile import build_profile

        return json.dumps(build_profile(case_id, client), indent=2)

    if name == "build_case_timeline":
        from acpia.timeline import build_timeline

        return _fmt_timeline(*build_timeline(case_id))

    if name == "list_other_cases":
        others = [c for c in events_store.list_cases() if c != case_id]
        return "Other cases: " + (", ".join(others) if others else "(none found)")

    if name == "correlate_cases":
        from acpia.correlate import correlate

        others = arguments.get("other_case_ids") or []
        if not others:
            return "No other case IDs provided to correlate against."
        result = correlate([case_id, *others], client)
        return _fmt_correlate(result)

    raise ValueError(f"unknown tool: {name}")
