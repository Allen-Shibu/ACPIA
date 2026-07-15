"""Compact a whole case into a small, CITED JSON profile.

This is the correlation building block: instead of a lossy prose summary (which
drops phone numbers and source links), we compress each case into structured
entities/identifiers/behaviours, every item pointing back to the memory it came
from. Two compact profiles can then be compared cheaply.

Citation trick: the LLM cites memories by integer index (0,1,2...); we map those
indices back to real memory IDs in code, and drop any out-of-range index — so a
citation can never be hallucinated.
"""

import json
import os
import re
import sys

from openai import OpenAI

from acpia.supermemory_client import SupermemoryClient


def _empty_profile(case: str) -> dict:
    return {"case": case, "people": [], "identifiers": [], "locations": [], "behaviors": []}


def _extract_with_retry(llm, model: str, messages: list, attempts: int = 2):
    """Call the model and parse its JSON, retrying on malformed output.

    Small local models intermittently emit invalid JSON (a stray/missing delimiter).
    A retry usually succeeds because generation is stochastic. Returns the parsed
    dict, or None if every attempt fails — the caller degrades gracefully rather
    than letting one bad response crash a whole multi-case run (e.g. correlate)."""
    last = None
    for _ in range(attempts):
        resp = llm.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"}
        )
        try:
            return extract_json(resp.choices[0].message.content)
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    print(f"[profile] LLM returned unparseable JSON after {attempts} attempts: {last}",
          file=sys.stderr)
    return None


def extract_json(content: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating <think> blocks and stray
    prose that small/reasoning local models emit even in JSON mode."""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    if not text:
        # Reasoning models (e.g. forced-thinking builds) can spend their whole token
        # budget thinking and return empty content. Name the cause instead of a
        # cryptic decode error. Fix: use a non-thinking model for structured calls.
        raise ValueError(
            "LLM returned empty content (no JSON). If using a local reasoning/thinking "
            "model, switch OPENAI_MODEL to a non-thinking one (e.g. qwen2.5:3b)."
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# /no_think: disable reasoning-token output on Qwen3.x (harmless to other models);
# a fixed JSON schema gains nothing from a chain of thought and it slows/derails small models.
_SCHEMA_INSTRUCTIONS = """\
/no_think
You are compacting one investigation case into a structured profile.
From the numbered evidence memories below, extract ONLY what is literally stated:

- people:      each distinct person -> {"name", "aliases": [...], "role": <one of: suspect, victim, witness, other, unknown>, "cites": [memory numbers]}
- identifiers: each concrete identifier -> {"type": <one of: phone, email, username, device, vehicle>, "value", "cites": [memory numbers]}
- locations:   each place -> {"name", "cites": [memory numbers]}
- behaviors:   notable actions/tactics -> {"desc", "cites": [memory numbers]}

Rules:
- Never invent a person, identifier, or detail not present in the text.
- For every "<one of: ...>" field, pick EXACTLY ONE value from the list. Never output the list itself or a slash-separated string like "victim|witness".
- Omit any item whose value is not literally in the text. Never emit placeholder values like "Not provided" or "Unknown location".
- "cites" must be the numbers of the memories that state that item.
- Return a single JSON object with keys: people, identifiers, locations, behaviors.
"""


def build_profile(case: str, client: SupermemoryClient | None = None) -> dict:
    """Return a compact, cited profile of the case (or an empty profile if no memories)."""
    client = client or SupermemoryClient()
    memories = client.fetch_all_memories(case)
    if not memories:
        return _empty_profile(case)

    index_to_id = {i: m["id"] for i, m in enumerate(memories)}
    numbered = "\n".join(f"{i}. {m['memory']}" for i, m in enumerate(memories))

    llm = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    raw = _extract_with_retry(
        llm, os.environ["OPENAI_MODEL"],
        [
            {"role": "system", "content": _SCHEMA_INSTRUCTIONS},
            {"role": "user", "content": f"Evidence memories:\n{numbered}"},
        ],
    )
    # Degrade gracefully: an unparseable response yields an empty profile for THIS
    # case, so correlate/timeline over other cases still complete.
    if raw is None:
        return _empty_profile(case)

    # Small models are inconsistent with shape: wrong key casing ("People"), a
    # section value that's a string instead of a list, or items that are bare
    # strings instead of objects. Normalize keys and skip anything malformed, so a
    # wrong-shape response degrades to fewer/no items rather than crashing.
    if not isinstance(raw, dict):
        print(f"[profile] LLM returned non-object JSON for case '{case}'; empty profile.",
              file=sys.stderr)
        return _empty_profile(case)
    raw = {k.lower(): v for k, v in raw.items()}
    key_field = {"people": "name", "identifiers": "value", "locations": "name", "behaviors": "desc"}
    profile = {"case": case}
    for section in ("people", "identifiers", "locations", "behaviors"):
        raw_items = raw.get(section)
        if not isinstance(raw_items, list):
            raw_items = []
        items = (_resolve_cites(item, index_to_id)
                 for item in raw_items if isinstance(item, dict))
        profile[section] = [it for it in items if str(it.get(key_field[section], "")).strip()]
    return profile


def _resolve_cites(item: dict, index_to_id: dict) -> dict:
    """Map integer cite indices to real memory IDs, dropping any out-of-range index."""
    cites = item.get("cites", []) or []
    resolved = []
    for c in cites:
        if isinstance(c, int) and c in index_to_id:
            resolved.append(index_to_id[c])
    item = dict(item)
    item["cites"] = resolved
    return item
