"""Compact a case into a cited PERSONS-OF-INTEREST brief.

This is the closest ACPIA gets to "who should we look at" — and it deliberately
stops one inch short. It lists each person with the concerning indicators the
evidence states about them, every indicator cited, ranked by how much evidence
concentrates on them. It NEVER declares a culprit, a score, or guilt: that
determination stays with the human investigator (see CLAUDE.md's core promise).

Same hallucination-proof citation trick as profile.py: the LLM cites memories by
integer index; code maps index->real id and drops out-of-range indices, so a
citation can't be invented. Persons/indicators with no surviving citation are
dropped — an unsourced lead is noise, not a lead.
"""

import os

from openai import OpenAI

from acpia.profile import _extract_with_retry, _resolve_cites
from acpia.supermemory_client import SupermemoryClient

# /no_think: no chain-of-thought for a fixed-schema extraction (see profile.py).
_SCHEMA = """/no_think
You are triaging ONE child-protection investigation case into LEADS for a human
investigator. From the numbered evidence memories below, list persons of interest.

For each distinct person named in the evidence, output:
  {"name", "role": <one of: suspect, victim, witness, other, unknown>,
   "indicators": [{"desc": "<one concerning behaviour or fact, literally stated>",
                   "cites": [memory numbers that state it]}]}

Rules:
- Include ONLY what is literally stated. Never invent a person or an indicator.
- Each indicator's "cites" are the numbers of the memories that state it.
- Do NOT rank, score, accuse, or declare anyone guilty. Just list people and
  their cited indicators; the human investigator draws the conclusion.
- Return a single JSON object: {"persons": [ ... ]}.
"""


def _shape_report(raw, index_to_id: dict, case: str) -> dict:
    """Pure: parse->filter->rank the LLM output. No network. Drops any person or
    indicator with no surviving citation, then ranks people by cited-indicator
    count (a crude evidence-concentration proxy, NOT a risk score)."""
    persons = []
    if isinstance(raw, dict):
        raw = {k.lower(): v for k, v in raw.items()}
        for p in raw.get("persons", []) or []:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "")).strip()
            if not name:
                continue
            indicators = []
            for ind in p.get("indicators", []) or []:
                if not isinstance(ind, dict):
                    continue
                ind = _resolve_cites(ind, index_to_id)
                if str(ind.get("desc", "")).strip() and ind["cites"]:
                    indicators.append({"desc": ind["desc"], "cites": ind["cites"]})
            if indicators:  # no cited indicator -> not a lead, drop the person
                persons.append({"name": name,
                                "role": str(p.get("role", "unknown")).strip() or "unknown",
                                "indicators": indicators})
    persons.sort(key=lambda p: len(p["indicators"]), reverse=True)
    return {"case": case, "persons": persons}


def build_report(case: str, client: SupermemoryClient | None = None) -> dict:
    """Return {case, persons:[{name, role, indicators:[{desc, cites}]}]}, ranked."""
    client = client or SupermemoryClient()
    memories = client.fetch_all_memories(case)
    if not memories:
        return {"case": case, "persons": []}

    index_to_id = {i: m["id"] for i, m in enumerate(memories)}
    numbered = "\n".join(f"{i}. {m['memory']}" for i, m in enumerate(memories))

    llm = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    raw = _extract_with_retry(
        llm, os.environ["OPENAI_MODEL"],
        [
            {"role": "system", "content": _SCHEMA},
            {"role": "user", "content": f"Evidence memories:\n{numbered}"},
        ],
    )
    return _shape_report(raw, index_to_id, case)


def _demo() -> None:
    """Self-check for the pure shaping logic — no network, no LLM."""
    idx = {0: "m0", 1: "m1", 2: "m2"}
    raw = {"persons": [
        {"name": "coolguy88", "role": "suspect", "indicators": [
            {"desc": "requested secrecy from parents", "cites": [0]},
            {"desc": "proposed an in-person meeting", "cites": [1, 99]},  # 99 out of range
        ]},
        {"name": "alexk_04", "role": "victim", "indicators": [
            {"desc": "recipient of the messages", "cites": [2]},
        ]},
        {"name": "Ghost", "role": "suspect", "indicators": [
            {"desc": "hallucinated with a bad cite", "cites": [42]},  # all out of range
        ]},
        {"name": "", "role": "other", "indicators": [{"desc": "no name", "cites": [0]}]},
    ]}
    rep = _shape_report(raw, idx, "c1")
    names = [p["name"] for p in rep["persons"]]
    assert names == ["coolguy88", "alexk_04"], names          # Ghost + nameless dropped
    assert rep["persons"][0]["indicators"][0]["cites"] == ["m0"]
    # out-of-range index stripped, valid one kept
    assert rep["persons"][0]["indicators"][1]["cites"] == ["m1"], rep["persons"][0]
    # ranked: coolguy88 (2 indicators) before alexk_04 (1)
    assert [len(p["indicators"]) for p in rep["persons"]] == [2, 1]
    # a person with only citeless indicators is never emitted
    assert "Ghost" not in names
    print("summarize self-check OK —", len(rep["persons"]), "persons of interest")


if __name__ == "__main__":
    _demo()
