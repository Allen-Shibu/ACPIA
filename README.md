# ACPIA

A CLI AI agent that turns digital evidence into cited, human-reviewable leads
for child-protection cases. Every output is lead-generation for human
review — **never a determination, verdict, or risk score.** Every claim cites
its source memory as `[memory:<id>]`, so the underlying evidence is always
traceable.

Built for the Localhost:6767 hackathon (Supermemory Local track, July 9–13
2026). See [`context.md`](context.md) for the problem this addresses and
[`CLAUDE.md`](CLAUDE.md) for the operational/dev guide.

## Why

Child-protection cases involve large volumes of evidence — chats, PDFs,
images, notes — spread across formats that are slow to search and hard to
correlate by hand. ACPIA ingests it all into a per-case memory store, then
lets you ask questions, build timelines, profile entities, and find links
across cases — with every claim traceable back to its source.

## Install

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run acpia run [./folder]                            # ingest a whole folder, then investigate (case = folder name)
uv run acpia ingest ./file.pdf ./chat.txt --case c1     # multi-file; txt/md/csv/json/pdf/images
uv run acpia ask "who is mentioned?" --case c1
uv run acpia investigate --case c1                      # interactive REPL
uv run acpia timeline --case c1 [--no-llm]
uv run acpia profile --case c1                          # cited JSON entity profile
uv run acpia summarize --case c1                        # cited persons-of-interest brief
uv run acpia correlate --case a --case b [--no-fuzzy]   # cross-case links
```

## How it works

Two backends:

- **Supermemory Local** (`localhost:6767`) — the evidence memory store. Each
  case gets its own `containerTag`; search is scoped to a single case, with
  no cross-case search.
- **An OpenRouter-hosted LLM** — the agent's reasoning/tool loop and all
  extraction.

Local structured events (`.acpia/<case>.events.json`) hold ground-truth
metadata (chat timestamps, image EXIF/GPS) alongside Supermemory's semantic
memories, so ground truth always outranks LLM prose when both exist.

Citations are non-negotiable: the LLM cites evidence by integer index, and
code maps index → real memory id, dropping anything out of range — this
makes fabricated citations structurally impossible.

Image handling reads EXIF metadata only — never the pixels or content of the
media.

## Project layout

| File | Responsibility |
|---|---|
| `cli.py` | Commands |
| `agent.py`, `tools.py` | LLM tool loop |
| `supermemory_client.py` | HTTP client for Supermemory Local |
| `extractors.py` | File → text extraction |
| `chat_parser.py`, `exif.py`, `events_store.py` | Ground-truth metadata |
| `dates.py`, `timeline.py` | Timeline reconstruction |
| `profile.py` | Case → cited JSON entity profile |
| `summarize.py` | Case → cited persons-of-interest brief |
| `correlate.py` | Cross-case linkage |

## Status

Done: ingest, ask, investigate, timeline, profile, correlate, summarize.
Next: entity/relationship graph. Deferred: Detroit-style branching HTML
timeline.
