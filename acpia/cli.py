import re
import pathlib

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from acpia import events_store
from acpia.agent import Agent
from acpia.chat_parser import looks_like_chat, parse_chat
from acpia.exif import extract_image_metadata
from acpia.extractors import IMAGE_SUFFIXES, UnsupportedFileType, extract_text
from acpia.supermemory_client import SupermemoryClient
from acpia.timeline import build_timeline

load_dotenv()
console = Console()


def _ingest_image(client: SupermemoryClient, case: str, path: pathlib.Path, vision: bool = False) -> None:
    """Images: extract EXIF metadata (no pixels) by default. With vision=True, ALSO run
    a LOCAL vision model over the pixels for a neutral description + OCR. Both are written
    to a clean labeled Supermemory doc (searchable, visible to profile/correlate)."""
    meta = extract_image_metadata(path)

    description = None
    if vision:
        from acpia.vision import CloudVisionRefused, describe_image
        try:
            description = describe_image(path)
        except CloudVisionRefused as e:
            console.print(f"[red]vision refused[/red] {path.name} — {e}")
            return

    if not any([meta["timestamp"], meta.get("gps"), meta.get("camera")]) and not description:
        console.print(f"[yellow]{path.name}[/yellow] — no EXIF metadata found")
        return
    if meta["timestamp"] or meta.get("gps") or meta.get("camera"):
        events_store.add_events(case, [events_store.make_image_event(case, path.name, meta)])

    lines = [f"Image evidence metadata (from EXIF headers).", f"Source file: {path.name}"]
    if meta["timestamp"]:
        lines.append(f"Capture timestamp: {meta['timestamp']}")
    if meta.get("camera"):
        lines.append(f"Device / camera: {meta['camera']}")
    if meta.get("gps"):
        g = meta["gps"]
        lines.append(f"GPS coordinates: latitude {g['lat']:.4f}, longitude {g['lon']:.4f}")
    if description:
        # AI-generated visual description — unverified lead, flagged for human review.
        lines.append(f"Visual description (LOCAL vision model, UNVERIFIED — verify against image):\n{description}")
    client.ingest(content="\n".join(lines), container_tag=case)

    bits = []
    if meta["timestamp"]:
        bits.append(f"captured {meta['timestamp']}")
    if meta.get("gps"):
        bits.append(f"GPS {meta['gps']['lat']:.4f},{meta['gps']['lon']:.4f}")
    if meta.get("camera"):
        bits.append(meta["camera"])
    console.print(f"[green]metadata[/green] {path.name} — {' · '.join(bits)}")


def _ingest_one(client: SupermemoryClient, source: str, case: str, wait: bool, vision: bool = False) -> None:
    path = pathlib.Path(source)

    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        _ingest_image(client, case, path, vision=vision)
        return

    if path.is_file():
        try:
            text = extract_text(path)
        except UnsupportedFileType as e:
            console.print(f"[red]skipped[/red] {e}")
            return
        # Provenance header so extracted facts stay traceable to their source file.
        content = f"[source file: {path.name}]\n\n{text}"
        label = path.name
        # Chat exports carry real per-message timestamps — capture them as
        # ground-truth structured events, in addition to semantic ingestion.
        if looks_like_chat(text):
            parsed = parse_chat(text)
            events = [events_store.make_chat_event(case, path.name, p) for p in parsed]
            n = events_store.add_events(case, events)
            console.print(f"[green]metadata[/green] {path.name} — {n} timestamped messages")
    else:
        content = source
        label = "inline text"

    doc_id = client.ingest(content=content, container_tag=case)
    console.print(f"[green]queued[/green] {label} → document {doc_id} (case '{case}')")

    if not wait:
        return
    with console.status(f"processing {label}..."):
        doc = client.wait_for_document(doc_id)
    if doc["status"] == "failed":
        console.print(f"[red]processing failed[/red] for {label} ({doc_id})")
        return
    memories = doc.get("memories", [])
    console.print(f"[green]done[/green] {label} — extracted {len(memories)} memories:")
    for m in memories:
        console.print(f"  [dim][memory:{m['id']}][/dim] {m['memory']}")


@click.group()
def cli():
    """ACPIA - AI investigation support agent."""


@cli.command()
@click.argument("sources", nargs=-1, required=True)
@click.option("--case", required=True, help="Case ID to tag this evidence under.")
@click.option("--wait/--no-wait", default=True, help="Wait for processing to finish.")
@click.option("--vision", is_flag=True, help="Run a LOCAL vision model over images (pixels stay local).")
def ingest(sources: tuple[str, ...], case: str, wait: bool, vision: bool):
    """Ingest evidence. SOURCES are file paths (txt, md, csv, json, pdf, ...) or raw text.

    Accepts multiple files at once, e.g.:  acpia ingest ./a.pdf ./b.txt --case c1
    """
    client = SupermemoryClient()
    for source in sources:
        _ingest_one(client, source, case, wait, vision=vision)


@cli.command()
@click.argument("question")
@click.option("--case", required=True, help="Case ID to search within.")
def ask(question: str, case: str):
    """Ask the agent a question about a case's evidence."""
    agent = Agent(case_id=case)
    with console.status("thinking..."):
        answer = agent.ask(question)
    console.print(Markdown(answer))


_SOURCE_STYLE = {
    "metadata": ("green", "●"),   # ground-truth from source metadata
    "explicit": ("cyan", "●"),    # explicit date in evidence text
    "inferred": ("yellow", "~"),  # LLM-inferred — verify against source
}


def _cites_markup(cite_ids) -> str:
    """Render memory citations as Rich markup. The literal '[' MUST be escaped as
    '\\[' — otherwise Rich parses '[memory:...]' as a (bogus) style tag and silently
    drops the whole citation. Citations are non-negotiable, so this must not vanish."""
    return " ".join(f"[dim]\\[memory:{c}][/dim]" for c in cite_ids)


def _banner() -> None:
    """Pixel-block startup banner. pagga font (pyfiglet) + cyan, matching the
    supermemory-server look. Purely cosmetic — swap font=/color to taste."""
    from pyfiglet import Figlet

    art = Figlet(font="pagga").renderText("ACPIA")
    console.print(f"[bold cyan]{art}[/bold cyan]", highlight=False)
    console.print("[dim]child-protection investigation assistant · leads for human review[/dim]\n")


def _clean_source(title: str | None) -> str:
    """Prefer the source filename from our provenance header; else a tidy title."""
    if not title:
        return ""
    m = re.search(r"\[source file:\s*([^\]]+)\]", title)
    if m:
        return m.group(1).strip()
    return " ".join(title.split())[:40]


@cli.command()
@click.option("--case", required=True, help="Case ID to build a timeline for.")
@click.option("--no-llm", is_flag=True, help="Skip LLM date inference for undated memories.")
def timeline(case: str, no_llm: bool):
    """Reconstruct a chronological timeline of events for a case."""
    with console.status("building timeline..."):
        dated, undated = build_timeline(case, use_llm=not no_llm)

    if not dated and not undated:
        console.print(f"[yellow]No events found for case '{case}'.[/yellow]")
        return

    console.print(f"\n[bold]Timeline — case '{case}'[/bold]")
    console.print(
        "[dim]● metadata (real)   ● explicit date   ~ inferred (verify)[/dim]\n"
    )
    for e in dated:
        color, mark = _SOURCE_STYLE.get(e["date_source"], ("white", "•"))
        date = e["date"].replace("T", " ")
        cite = f" {_cites_markup([e['citation']])}" if e["citation"] else ""
        src = f" [dim]({_clean_source(e['source'])})[/dim]" if e["source"] else ""
        text = " ".join(e["text"].split())
        console.print(f"[{color}]{mark} {date}[/{color}]  {text}{cite}{src}")

    if undated:
        console.print(f"\n[bold yellow]Undated — needs review ({len(undated)})[/bold yellow]")
        for e in undated:
            cite = f" {_cites_markup([e['citation']])}" if e["citation"] else ""
            console.print(f"  • {e['text']}{cite}")
    console.print(
        "\n[dim]Draft timeline for investigator review. Verify inferred (~) dates "
        "against source evidence.[/dim]"
    )


@cli.command()
@click.option("--case", required=True, help="Case ID to profile.")
def profile(case: str):
    """Compact a case into a cited entity/identifier profile (JSON)."""
    from acpia.profile import build_profile

    with console.status("compacting case..."):
        prof = build_profile(case)
    console.print_json(data=prof)


def _render_links(title: str, links: list, verify: bool) -> None:
    color = "yellow" if verify else "green"
    mark = "~" if verify else "●"
    console.print(f"\n[bold {color}]{title} ({len(links)})[/bold {color}]")
    for link in links:
        console.print(f"[{color}]{mark} {link['value']}[/{color}]")
        for m in link["members"]:
            cites = _cites_markup(m["cites"])
            console.print(f"    [dim]case[/dim] {m['case']}: {m['display']}  {cites}")


@cli.command()
@click.option("--case", "cases", required=True, multiple=True,
              help="Case ID (repeat for each case, e.g. --case a --case b).")
@click.option("--no-fuzzy", is_flag=True, help="Skip the LLM soft-match pass (exact only).")
def correlate(cases: tuple[str, ...], no_fuzzy: bool):
    """Find people, identifiers, and locations shared across two or more cases."""
    from acpia.correlate import correlate as run_correlate

    if len(cases) < 2:
        console.print("[red]correlate needs at least two --case values.[/red]")
        return
    with console.status("profiling and correlating cases..."):
        result = run_correlate(list(cases), fuzzy=not no_fuzzy)

    console.print(f"\n[bold]Cross-case links — {', '.join(cases)}[/bold]")
    console.print("[dim]● exact match (hard link)   ~ LLM soft match (verify)[/dim]")
    if result["exact"]:
        _render_links("Exact matches", result["exact"], verify=False)
    if result["fuzzy"]:
        _render_links("Soft matches — needs review", result["fuzzy"], verify=True)
    if not result["exact"] and not result["fuzzy"]:
        console.print("\n[yellow]No shared entities found across these cases.[/yellow]")
    console.print(
        "\n[dim]Lead-generation for investigator review. A shared entity is a lead to "
        "verify, not a determination that the cases involve the same person.[/dim]"
    )


@cli.command()
@click.option("--case", required=True, help="Case ID to summarize.")
def summarize(case: str):
    """Cited persons-of-interest brief — leads for review, NOT a determination of guilt."""
    from acpia.summarize import build_report

    with console.status("summarizing case..."):
        rep = build_report(case)
    persons = rep["persons"]
    if not persons:
        console.print(f"[yellow]No cited persons of interest found for case '{case}'.[/yellow]")
        return

    console.print(f"\n[bold]Persons of Interest — case '{case}'[/bold]")
    console.print("[dim]▲ evidence concentrates here (a lead, NOT a finding of guilt)[/dim]\n")
    top = len(persons[0]["indicators"])  # persons is rank-sorted; ties share the mark
    for p in persons:
        concentrated = len(p["indicators"]) == top and top > 1
        color, mark = ("yellow", "▲") if concentrated else ("white", "·")
        console.print(f"[{color}]{mark} {p['name']}[/{color}]  "
                      f"[dim]{p['role']} — {len(p['indicators'])} indicator(s)[/dim]")
        for ind in p["indicators"]:
            cites = _cites_markup(ind["cites"])
            console.print(f"    • {ind['desc']}  {cites}")
    console.print(
        "\n[dim]Evidence concentration is a lead to investigate, not a finding of guilt. "
        "The determination is the investigator's. Verify every cited item against source "
        "evidence.[/dim]"
    )


def _run_repl(agent: Agent, case: str, brief: bool = True) -> None:
    """Shared interactive loop: optional opening auto-analysis, then Q&A until Ctrl+D."""
    console.print(f"investigating case '{case}'. Ctrl+D to exit.\n")
    if brief:
        with console.status("analyzing case (profile, timeline, cross-case links)..."):
            briefing = agent.orient()
        console.print(Markdown(briefing))
        console.print()
    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]")
        except EOFError:
            break
        if not question.strip():
            continue
        with console.status("thinking..."):
            answer = agent.ask(question)
        console.print(Markdown(answer))
        console.print()


@cli.command()
@click.option("--case", required=True, help="Case ID to investigate.")
@click.option("--no-brief", is_flag=True, help="Skip the automatic opening case analysis.")
def investigate(case: str, no_brief: bool):
    """Interactive REPL for a case. Auto-analyzes the case on entry."""
    _banner()
    _run_repl(Agent(case_id=case), case, brief=not no_brief)


@cli.command()
@click.argument("folder", default=".")
@click.option("--case", default=None, help="Case ID (default: the folder's name).")
@click.option("--vision", is_flag=True, help="Run a LOCAL vision model over images.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt (unattended runs).")
def run(folder: str, case: str | None, vision: bool, yes: bool):
    """Analyze an evidence FOLDER end to end: ingest every supported file, then open
    the investigation. FOLDER defaults to the current directory; case defaults to its name.
    """
    from acpia.extractors import IMAGE_SUFFIXES, PDF_SUFFIXES, TEXT_SUFFIXES

    _banner()
    root = pathlib.Path(folder).resolve()
    case = case or root.name
    supported = TEXT_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES
    files = sorted(p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in supported)
    if not files:
        console.print(f"[yellow]No supported evidence files in {root}.[/yellow]")
        return

    console.print(f"[bold]Case '{case}'[/bold] — {len(files)} evidence file(s) in {root}:")
    for p in files:
        console.print(f"  • {p.name}")
    # Ask the officer before ingesting — ingestion sends evidence text to the memory store.
    if not (yes or click.confirm(f"\nIngest and investigate case '{case}'?", default=True)):
        return

    client = SupermemoryClient()
    ingested = queued = 0
    for p in files:
        try:
            _ingest_one(client, str(p), case, wait=True, vision=vision)
            ingested += 1
        except TimeoutError:
            # Doc was submitted; only Supermemory's slow local indexing didn't confirm
            # in time (known gotcha). It's in the store — proceed, it'll index shortly.
            console.print(f"[yellow]{p.name}[/yellow] — queued, still indexing")
            queued += 1
        except Exception as e:
            # A real failure (unreadable/unsupported file) shouldn't kill the batch.
            console.print(f"[red]skipped {p.name}[/red] — {type(e).__name__}: {e}")
    console.print()
    if not ingested and not queued:
        console.print("[yellow]Nothing ingested — aborting.[/yellow]")
        return
    if queued and not ingested:
        console.print("[dim]All evidence is still indexing; the opening analysis may lag "
                      "until it completes.[/dim]\n")
    _run_repl(Agent(case_id=case), case, brief=True)


if __name__ == "__main__":
    cli()
