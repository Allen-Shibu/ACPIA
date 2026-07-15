"""ACPIA eval harness — scores a run against a known answer key.

Two tiers:
  Tier A  deterministic ground-truth checks. No network, no LLM. Tests the
          reliability-critical logic (chat timestamps, trust-tier dates, EXIF
          fidelity, cross-case exact matching, citation hallucination-proofing).
          This is the scoreboard you tune prompts/models against.
  Tier B  LLM extraction quality (profile lead-recall + cross-case correlate).
          Uses a stub memory store so it exercises the real extraction/correlation
          LLM logic WITHOUT depending on the local Supermemory indexing (which is
          slow/unreliable on weak hardware). Enable with:  --llm

Run:  uv run python evals/run.py [--llm]
"""

import json
import pathlib
import re
import sys

from dotenv import load_dotenv

HERE = pathlib.Path(__file__).parent
load_dotenv(HERE.parent / ".env")

from acpia import correlate as C
from acpia.chat_parser import looks_like_chat, parse_chat
from acpia.dates import _verbatim_date, resolve_dates
from acpia.exif import extract_image_metadata
from acpia.profile import _resolve_cites, build_profile

FIX = HERE / "fixtures"
KEY = json.loads((HERE / "answer_key.json").read_text())

_results: list[tuple[str, bool, str]] = []


def check(name: str, passed, detail: str = ""):
    _results.append((name, bool(passed), detail))


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# --------------------------------------------------------------------------- #
# Tier A — deterministic (no LLM)
# --------------------------------------------------------------------------- #
def tier_a():
    lk = KEY["lakeside"]

    # -- chat parsing: real per-message timestamps (ground truth) --
    chat = (FIX / "case-lakeside" / "chat_lakeside.txt").read_text()
    ev = parse_chat(chat)
    check("chat/detected", looks_like_chat(chat))
    check("chat/message-count", len(ev) == lk["chat_message_count"], f"got {len(ev)}")
    check("chat/senders", {e["sender"] for e in ev} == set(lk["senders"]),
          str({e["sender"] for e in ev}))
    check("chat/all-timestamped", all(e["timestamp"] for e in ev))
    check("chat/first-timestamp", ev[0]["timestamp"] == lk["first_timestamp"],
          ev[0]["timestamp"])

    # -- date trust tiers --
    tip = (FIX / "case-lakeside" / "parent_tip.txt").read_text()
    check("date/explicit-iso-found", _verbatim_date(tip) == lk["explicit_date"],
          str(_verbatim_date(tip)))
    check("date/relative-not-explicit", _verbatim_date("meet this saturday at 4pm") is None)
    mems = [
        {"id": "a", "memory": f"On {lk['explicit_date']} a parent reported coach Dave.", "metadata": {}},
        {"id": "b", "memory": "coach_dave asked Jamie to meet this Saturday.", "metadata": {}},
    ]
    by = {m["id"]: m for m in resolve_dates(mems, use_llm=False)}
    check("date/explicit-tier",
          by["a"]["date"] == lk["explicit_date"] and by["a"]["date_source"] == "explicit")
    check("date/relative-undated-without-llm",
          by["b"]["date"] is None and by["b"]["date_source"] is None)

    # -- EXIF fidelity (generate an image, read it back) --
    img = _make_exif_image()
    meta = extract_image_metadata(img)
    ex = lk["exif"]
    check("exif/timestamp", meta["timestamp"] == ex["timestamp"], str(meta["timestamp"]))
    check("exif/gps-precision",
          bool(meta["gps"]) and abs(meta["gps"]["lat"] - ex["lat"]) < 1e-4
          and abs(meta["gps"]["lon"] - ex["lon"]) < 1e-4, str(meta.get("gps")))
    check("exif/camera", meta["camera"] == ex["camera"], str(meta["camera"]))

    # -- cross-case exact matching (the hard link investigators act on) --
    check("correlate/phone-normalizes-equal",
          C._norm("+1-555-0148") == C._norm("+1 (555) 0148"))
    pl = {"case": "lakeside",
          "people": [{"name": "coach_dave", "aliases": ["Dave"], "cites": ["l0"]}],
          "identifiers": [{"type": "phone", "value": "+1-555-0148", "cites": ["l4"]}],
          "locations": [], "behaviors": []}
    pp = {"case": "parkview",
          "people": [{"name": "coach_dave", "aliases": [], "cites": ["p0"]}],
          "identifiers": [{"type": "phone", "value": "+1 (555) 0148", "cites": ["p1"]}],
          "locations": [], "behaviors": []}
    exact = C._group_exact(C._entity_keys(pl) + C._entity_keys(pp))
    vals = {l["value"] for l in exact}
    check("correlate/username-links-cross-case", any("coach_dave" in v for v in vals), str(vals))
    check("correlate/phone-links-cross-case",
          any("5550148" in digits(v) for v in vals), str(vals))
    # every exact link must carry cites from BOTH cases
    ok_cites = all({m["case"] for m in l["members"]} == {"lakeside", "parkview"}
                   and all(m["cites"] for m in l["members"]) for l in exact)
    check("correlate/links-carry-both-sides-cites", ok_cites)

    # -- citation hallucination-proofing --
    it = _resolve_cites({"name": "x", "cites": [0, 2, 99, "bad"]}, {0: "m0", 2: "m2"})
    check("citation/drops-out-of-range-and-noninteger", it["cites"] == ["m0", "m2"],
          str(it["cites"]))

    # -- robustness: a malformed LLM response must degrade, not crash --
    # (regression for the JSONDecodeError that Tier B surfaced in correlate)
    import acpia.profile as P
    for label, fake in (("malformed-json", _bad_json_openai),
                        ("wrong-shape-json", _wrong_shape_openai)):
        orig = P.OpenAI
        P.OpenAI = fake
        try:
            prof = P.build_profile("lakeside", StubClient())
            check(f"robust/{label}-degrades-to-empty",
                  prof == P._empty_profile("lakeside"), str(prof))
        except Exception as e:
            check(f"robust/{label}-degrades-to-empty", False, f"raised {type(e).__name__}")
        finally:
            P.OpenAI = orig


def _bad_json_openai(*a, **k):
    """A fake OpenAI client whose completions always return unparseable JSON."""
    class _Msg:  # noqa
        content = '{"people": [ {"name": "x" ,, ] }'
    class _Resp:  # noqa
        choices = [type("C", (), {"message": _Msg()})()]
    class _Client:  # noqa
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    return _Resp()
    return _Client()


def _wrong_shape_openai(*a, **k):
    """Valid JSON, wrong shape: items are bare strings, a section is a string."""
    return _canned_openai('{"people": ["coach_dave", "Jamie"], "identifiers": "none"}')


def _canned_openai(content: str):
    class _Msg:  # noqa
        pass
    _Msg.content = content
    class _Resp:  # noqa
        choices = [type("C", (), {"message": _Msg()})()]
    class _Client:  # noqa
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    return _Resp()
    return _Client()


def _make_exif_image() -> pathlib.Path:
    import piexif
    from PIL import Image
    ex = KEY["lakeside"]["exif"]

    def dms(deg):
        deg = abs(deg); d = int(deg); m = int((deg - d) * 60); s = round((deg - d - m / 60) * 3600, 4)
        return ((d, 1), (m, 1), (int(s * 100), 100))

    gps = {piexif.GPSIFD.GPSLatitudeRef: "N", piexif.GPSIFD.GPSLatitude: dms(ex["lat"]),
           piexif.GPSIFD.GPSLongitudeRef: "W", piexif.GPSIFD.GPSLongitude: dms(ex["lon"])}
    exif = {"0th": {piexif.ImageIFD.Make: "Apple", piexif.ImageIFD.Model: "iPhone 13"},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: ex["timestamp"].replace("-", ":").replace("T", " ")},
            "GPS": gps, "1st": {}, "thumbnail": None}
    out = FIX / "case-lakeside" / "scene.jpg"
    Image.new("RGB", (48, 48), (150, 160, 170)).save(out, exif=piexif.dump(exif))
    return out


# --------------------------------------------------------------------------- #
# Tier B — LLM extraction quality (stubbed memory store, real LLM)
# --------------------------------------------------------------------------- #
# Simulated Supermemory memories (what indexing WOULD extract), so Tier B tests
# the extraction/correlation LLM logic without the flaky local indexer.
_STUB_MEMS = {
    "lakeside": [
        "Dave, who uses the handle coach_dave, is a swim coach who has been privately messaging Jamie.",
        "coach_dave told Jamie to keep their chats secret and not tell anyone.",
        "coach_dave offered Jamie a gift of new swimming goggles.",
        "coach_dave asked Jamie to meet at the Lakeside Rec Center this Saturday at 4 PM.",
        "The phone number +1-555-0148 belongs to coach_dave.",
        "On 2026-06-03 a parent reported swim coach Dave for privately messaging their child Jamie.",
    ],
    "parkview": [
        "A user with the handle coach_dave was reported for contacting minors through a youth sports app in the Parkview district.",
        "The phone number +1 (555) 0148 was associated with the coach_dave account.",
        "The victim in the Parkview report is a child known as Sam.",
    ],
}


class StubClient:
    """Stand-in for SupermemoryClient.fetch_all_memories — returns canned memories
    per case so Tier B never touches the network/indexer."""
    def fetch_all_memories(self, case: str) -> list[dict]:
        return [{"id": f"{case[:2]}{i}", "memory": t, "metadata": {}}
                for i, t in enumerate(_STUB_MEMS.get(case, []))]


def _flat_values(profile: dict) -> str:
    """All extracted entity strings, lowercased, for substring lead-recall."""
    parts = []
    for p in profile.get("people", []):
        parts += [p.get("name", "")] + list(p.get("aliases") or [])
    parts += [i.get("value", "") for i in profile.get("identifiers", [])]
    parts += [l.get("name", "") for l in profile.get("locations", [])]
    parts += [b.get("desc", "") for b in profile.get("behaviors", [])]
    return " || ".join(parts).lower()


def _found(lead: str, blob: str) -> bool:
    # phone leads are all-digits in the key; match on digit-substring
    if lead.isdigit():
        return lead in digits(blob)
    return lead.lower() in blob


def tier_b():
    stub = StubClient()
    print("\n[Tier B] LLM extraction — this calls the local model; slow on CPU.\n")

    # profile lead-recall on lakeside
    prof = build_profile("lakeside", stub)
    blob = _flat_values(prof)
    lk = KEY["lakeside"]
    leads = lk["people"] + lk["identifiers"] + lk["locations"]
    hits = [l for l in leads if _found(l, blob)]
    recall = len(hits) / len(leads)
    print(f"  profile lead-recall: {len(hits)}/{len(leads)} = {recall:.0%}")
    for l in leads:
        print(f"    {'FOUND ' if _found(l, blob) else 'MISS  '} {l}")
    check("profileB/lead-recall>=0.8", recall >= 0.8, f"{recall:.0%}")

    # real cross-case correlate via the same stub (exact tier, no fuzzy LLM)
    result = C.correlate(["lakeside", "parkview"], client=stub, fuzzy=False)
    exact = result["exact"]
    vals = {l["value"] for l in exact}
    print(f"  correlate exact links: {sorted(vals)}")
    # every exact link must genuinely span both cases
    all_cross = all({m["case"] for m in l["members"]} == {"lakeside", "parkview"} for l in exact)
    check("correlateB/links-span-both-cases", bool(exact) and all_cross, str(vals))
    # the phone (differently formatted in each case) links -> normalization worked
    check("correlateB/phone-linked", any("5550148" in digits(v) for v in vals), str(vals))
    # a person links across cases; the ONLY shared person-key is the coach_dave
    # handle, so a person link existing == the username drove the match (display
    # value is the person's name, not the handle).
    person_linked = any(not digits(v) and "phone" not in v.lower() for v in vals)
    check("correlateB/person-linked-via-handle", person_linked, str(vals))


# --------------------------------------------------------------------------- #
def main():
    run_llm = "--llm" in sys.argv
    tier_a()
    if run_llm:
        try:
            tier_b()
        except Exception as e:
            print(f"\n[Tier B ERROR] {type(e).__name__}: {e}")
            check("tierB/completed", False, str(e)[:120])

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    for name, ok, detail in _results:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail and not ok:
            line += f"   ({detail})"
        print(line)
    print("=" * 60)
    print(f"  {passed}/{len(_results)} checks passed")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
