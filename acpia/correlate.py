"""Cross-case linkage: find entities shared across two or more cases.

Supermemory search is scoped to a single case (containerTag), so there is no
native way to ask "does this device appear in another case?". Correlate answers
it by profiling each case (profile.build_profile) and comparing the compact,
cited entity lists.

Two tiers, mirroring the trust tiers used elsewhere:
- EXACT: normalized string equality on identifiers / names / locations. A shared
  username or device IS a shared username — provably correct, no LLM, the hard
  link investigators actually act on.
- FUZZY: an LLM clusters the leftover people/locations that did NOT exact-match
  ("Mike" vs "Michael"). Flagged "verify" because an LLM can wrongly merge two
  different people — a serious false lead in this domain, so it's a suggestion
  for human review, never an assertion.

Every link carries the real memory-id cites from BOTH sides so an investigator
can pull the source evidence on each case. Output is lead-generation for human
review, never a determination that two cases involve the same person.
"""

import json
import os
import re

from openai import OpenAI

from acpia.profile import build_profile, extract_json
from acpia.supermemory_client import SupermemoryClient


def _norm(s: str | None) -> str:
    """Lowercase, strip everything but letters/digits. Collapses punctuation and
    spacing so '+1 (555) 123-4567' and '15551234567' compare equal."""
    # ponytail: naive normalize. A US number with/without country code ('1' prefix)
    #   won't match; add phone-aware last-10-digit compare if that shows up for real.
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _entity_keys(profile: dict) -> list[tuple[str, str, str, list]]:
    """Flatten one profile into (key, case, display, cites) rows for matching.
    A person contributes one row per name/alias so an alias in case A can link to
    a real name in case B. Behaviors are prose, not linkable — excluded."""
    case = profile["case"]
    rows: list[tuple[str, str, str, list]] = []
    for p in profile.get("people", []):
        names = [p.get("name")] + list(p.get("aliases") or [])
        for n in names:
            if _norm(n):
                rows.append((_norm(n), case, p.get("name") or n, p.get("cites", [])))
    for i in profile.get("identifiers", []):
        if _norm(i.get("value")):
            display = f"{i.get('type', '?')}: {i.get('value')}"
            rows.append((_norm(i["value"]), case, display, i.get("cites", [])))
    for loc in profile.get("locations", []):
        if _norm(loc.get("name")):
            rows.append((_norm(loc["name"]), case, loc["name"], loc.get("cites", [])))
    return rows


def _group_exact(rows: list[tuple[str, str, str, list]]) -> list[dict]:
    """Group rows by normalized key; keep only keys spanning >1 case."""
    by_key: dict[str, list] = {}
    for key, case, display, cites in rows:
        by_key.setdefault(key, []).append({"case": case, "display": display, "cites": cites})
    links = []
    for key, members in by_key.items():
        if len({m["case"] for m in members}) > 1:
            links.append({"value": members[0]["display"], "members": members})
    return links


def _exact_singletons(profile: dict, matched_keys: set[str]) -> list[dict]:
    """People/locations from one profile whose key did NOT exact-match — fuzzy candidates."""
    out = []
    for p in profile.get("people", []):
        keys = {_norm(n) for n in [p.get("name")] + list(p.get("aliases") or []) if _norm(n)}
        if keys and keys.isdisjoint(matched_keys):
            out.append({"case": profile["case"], "kind": "person",
                        "value": p.get("name"), "cites": p.get("cites", [])})
    for loc in profile.get("locations", []):
        if _norm(loc.get("name")) and _norm(loc["name"]) not in matched_keys:
            out.append({"case": profile["case"], "kind": "location",
                        "value": loc["name"], "cites": loc.get("cites", [])})
    return out


_FUZZY_INSTRUCTIONS = """\
/no_think
You are helping an investigator spot when two DIFFERENT cases may involve the same \
real-world person or place, despite different spellings (e.g. "Mike" vs "Michael", \
"the docks" vs "Riverside Marina").

Below are candidate entities, each numbered, each tagged with its case. Group ONLY \
candidates that plausibly refer to the SAME real-world entity AND come from at least \
two DIFFERENT cases. Be conservative — when unsure, do not group. Never merge two \
clearly distinct people.

Return JSON: {"groups": [[numbers], ...]}. Each inner list is the numbers of \
candidates you believe are the same entity. Omit anything you would not group.
"""


def _fuzzy_link(candidates: list[dict]) -> list[dict]:
    """LLM clusters leftover candidates across cases. Index->candidate mapped in code;
    out-of-range indices dropped (hallucination-proof, same trick as profile.py)."""
    if len(candidates) < 2:
        return []
    numbered = "\n".join(
        f"{i}. [case {c['case']}] ({c['kind']}) {c['value']}" for i, c in enumerate(candidates)
    )
    llm = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    resp = llm.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": _FUZZY_INSTRUCTIONS},
            {"role": "user", "content": f"Candidates:\n{numbered}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = extract_json(resp.choices[0].message.content)
    links = []
    for group in raw.get("groups", []) or []:
        idxs = [n for n in group if isinstance(n, int) and 0 <= n < len(candidates)]
        members = [candidates[n] for n in idxs]
        if len({m["case"] for m in members}) > 1:  # must still span >1 case
            links.append({
                "value": members[0]["value"],
                "members": [{"case": m["case"], "display": m["value"], "cites": m["cites"]}
                            for m in members],
            })
    return links


def correlate(cases: list[str], client: SupermemoryClient | None = None,
              fuzzy: bool = True) -> dict:
    """Profile every case and return exact + fuzzy cross-case entity links."""
    client = client or SupermemoryClient()
    profiles = [build_profile(c, client) for c in cases]

    rows = [row for p in profiles for row in _entity_keys(p)]
    exact = _group_exact(rows)
    matched_keys = {_norm(m["display"].split(": ")[-1]) for link in exact for m in link["members"]}

    fuzzy_links: list[dict] = []
    if fuzzy:
        candidates = [c for p in profiles for c in _exact_singletons(p, matched_keys)]
        fuzzy_links = _fuzzy_link(candidates)

    return {"cases": cases, "exact": exact, "fuzzy": fuzzy_links}


def _demo() -> None:
    """Self-check for the pure exact-match logic — no network, no LLM."""
    a = {"case": "A",
         "people": [{"name": "MikeD_99", "aliases": ["Mike"], "cites": ["a1"]}],
         "identifiers": [{"type": "phone", "value": "(555) 123-4567", "cites": ["a2"]}],
         "locations": [{"name": "Riverside Marina", "cites": ["a3"]}]}
    b = {"case": "B",
         "people": [{"name": "unrelated", "aliases": ["miked_99"], "cites": ["b1"]},
                    {"name": "SoloWitness", "aliases": [], "cites": ["b4"]}],
         "identifiers": [{"type": "phone", "value": "555-123-4567 ", "cites": ["b2"]}],
         "locations": [{"name": "riverside  marina", "cites": ["b3"]}]}
    rows = _entity_keys(a) + _entity_keys(b)
    exact = _group_exact(rows)
    values = {link["value"] for link in exact}
    # alias->name link, normalized phone, and normalized location all cross A<->B
    assert "MikeD_99" in values, values
    assert any("555" in v for v in values), values
    assert "Riverside Marina" in values, values
    # cites from both sides preserved on the person link
    person = next(l for l in exact if l["value"] == "MikeD_99")
    assert {m["case"] for m in person["members"]} == {"A", "B"}
    assert ["a1"] in [m["cites"] for m in person["members"]]
    # a case with a unique-only entity is a fuzzy candidate, not an exact link
    matched = {_norm(m["display"].split(": ")[-1]) for l in exact for m in l["members"]}
    singles = _exact_singletons(b, matched)
    assert any(s["value"] == "SoloWitness" for s in singles), singles
    print("correlate self-check OK —", len(exact), "exact links")


if __name__ == "__main__":
    _demo()
