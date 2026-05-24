"""Rich-based TUI replacement for FindTheMag2's print_table().

Opt-in via ``USE_RICH_TUI = True`` in config.py. Falls back transparently
to the legacy ASCII print_table if rich is not installed or the flag is off.

Design notes:
    * One ``rich.live.Live`` region is created lazily on first ``render()``
      call and kept open for the process lifetime. Subsequent calls update
      the region in place — no clear-screen, no flicker.
    * print() output from the rest of FTM2 (BOINC RPC errors, FTM startup
      banners, etc.) scrolls above the live area. That's acceptable noise;
      keeping logs visible is more useful than capturing them.
    * Column set is identical to the legacy table; only the rendering is
      replaced. New v2 columns (V2_µCR, V2_σCR, V2_EMAG) get their own
      highlight styling.
    * No interactive keys — pure rendering. Sort key is configurable via
      ``TUI_SORT_BY`` in config.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

# rich is an optional dep; importing it lazily so legacy mode still works
# without it installed.
try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


# Ordered (key, header, justify) — first column is always Project.
_COLUMNS: List[Tuple[str, str, str]] = [
    ("HOURSOFF",    "HOff",    "right"),
    ("TASKS",       "Tasks",   "right"),
    ("WTIME",       "Wall",    "right"),
    ("CPUTIME",     "CPU",     "right"),
    ("CREDIT/HR",   "Cr/Hr",   "right"),
    ("MAG/HR",      "Mag/Hr",  "right"),
    ("V2_µCR",      "v2 µCR",  "right"),
    ("V2_σCR",      "v2 σCR",  "right"),
    ("V2_EMAG",     "v2 EMag", "right"),
    ("WEIGHT",      "Weight",  "right"),
    ("GRC/HR",      "GRC/Hr",  "right"),
    ("GRC/DAY",     "GRC/Day", "right"),
    ("USD/DAY R/P", "$ R/P",   "right"),
]


_LIVE_SINGLETON: Optional["Live"] = None
_CONSOLE_SINGLETON: Optional["Console"] = None
_LEGEND_PRINTED: bool = False


def is_available() -> bool:
    """Whether rich is importable in the current environment."""
    return _RICH_AVAILABLE


def _ensure_live() -> "Live":
    """Lazy-init a Live region; kept open for process lifetime."""
    global _LIVE_SINGLETON, _CONSOLE_SINGLETON, _LEGEND_PRINTED
    if _LIVE_SINGLETON is None:
        _CONSOLE_SINGLETON = Console()
        if not _LEGEND_PRINTED:
            _print_legend(_CONSOLE_SINGLETON)
            _LEGEND_PRINTED = True
        _LIVE_SINGLETON = Live(
            console=_CONSOLE_SINGLETON,
            refresh_per_second=4,
            screen=False,
            transient=False,
        )
        _LIVE_SINGLETON.start()
    return _LIVE_SINGLETON


def _print_legend(console: "Console") -> None:
    """One-shot legend printed above the Live area before it opens."""
    console.print()
    console.rule("[bold magenta]FindTheMag · v2-algorithm")
    console.print(
        "[dim]"
        "HOff: hours off target ·  Wall/CPU: aggregate wall- and CPU-hours ·  "
        "Cr/Hr: observed credit per hour ·  Mag/Hr: expected magnitude per hour\n"
        "v2 µCR: posterior mean credit/hr (analytic) ·  v2 σCR: posterior stdev "
        "— larger means v2 is less confident and will explore more\n"
        "v2 EMag: posterior mean × smoothed mag/credit ratio ·  Weight: BOINC "
        "resource share (0-1000) — 1 is a 'kept attached' placeholder\n"
        "GRC/Hr|Day: gridcoin earnings projection ·  $ R/P: USD revenue/profit "
        "per day, factoring HOST_POWER_USAGE × LOCAL_KWH"
        "[/dim]"
    )
    console.print()


def stop() -> None:
    """Tear down the Live region cleanly (call on Ctrl-C / shutdown)."""
    global _LIVE_SINGLETON
    if _LIVE_SINGLETON is not None:
        try:
            _LIVE_SINGLETON.stop()
        except Exception:
            pass
        _LIVE_SINGLETON = None


def _format_cell(value: Any, col_key: str) -> "Text":
    """Apply per-column styling. Cells with empty/zero values are dimmed."""
    s = "" if value is None else str(value)
    if s in ("", "0", "0.0", "0.00", "0.000"):
        return Text(s, style="dim")
    if col_key == "WEIGHT":
        try:
            v = float(s)
        except ValueError:
            return Text(s)
        if v >= 200:
            return Text(s, style="bold green")
        if v >= 50:
            return Text(s, style="green")
        if v <= 1:
            return Text(s, style="dim yellow")
        return Text(s, style="yellow")
    if col_key == "V2_σCR":
        return Text(s, style="cyan")
    if col_key == "V2_µCR":
        return Text(s, style="bright_white")
    if col_key == "V2_EMAG":
        return Text(s, style="bright_magenta")
    if col_key == "USD/DAY R/P":
        # Negative profit second half → red
        if "-" in s.split("/")[-1]:
            return Text(s, style="red")
    return Text(s)


def _project_label(project_url: str) -> str:
    """Shorten a project URL to a readable label, max 32 chars.

    Some BOINC project URLs put the meaningful name in the hostname and
    leave the last path segment as just "BOINC" (e.g. Asteroids@home is
    canonically ASTEROIDSATHOME.NET/BOINC). Picking the last segment for
    those gives a meaningless "boinc" label; fall back to the hostname.
    """
    s = project_url.rstrip("/")
    if "://" in s:
        s = s.split("://", 1)[1]
    parts = s.split("/")
    host = parts[0]
    path_parts = [p for p in parts[1:] if p]
    last = path_parts[-1].lower() if path_parts else ""
    if last and last not in ("boinc",):
        label = last
    else:
        label = host.lower()
    if len(label) > 32:
        label = label[:30] + "…"
    return label


def _sort_rows(
    rows: List[Tuple[str, Dict[str, str]]], sort_by: str
) -> List[Tuple[str, Dict[str, str]]]:
    def keyfn(r: Tuple[str, Dict[str, str]]) -> float:
        raw = r[1].get(sort_by, "0") or "0"
        try:
            return -float(raw)
        except (ValueError, TypeError):
            return 0.0
    return sorted(rows, key=keyfn)


def render(
    table_dict: Mapping[str, Mapping[str, str]],
    *,
    sleep_reason: str = "",
    status: str = "",
    grc_price: Optional[float] = None,
    v2_top3: Optional[List[Tuple[str, float]]] = None,
    sort_by: str = "WEIGHT",
    dev_status: bool = False,
    avg_mag_per_hr: Optional[float] = None,
    hours_user_vs_dev: Optional[Tuple[float, float]] = None,
) -> None:
    """Refresh the TUI in-place.

    Drop-in replacement for the body of ``print_table()`` when
    ``config.USE_RICH_TUI`` is true. Safe to call from a non-TUI context
    (no-op if rich is missing).
    """
    if not _RICH_AVAILABLE:
        return
    live = _ensure_live()

    tbl = Table(
        header_style="bold magenta",
        show_lines=False,
        expand=True,
        pad_edge=False,
        padding=(0, 1),
    )
    tbl.add_column("Project", no_wrap=True, style="bold", min_width=18)
    for key, header, justify in _COLUMNS:
        tbl.add_column(header, justify=justify, no_wrap=True)

    rows = list(table_dict.items())
    rows = _sort_rows(rows, sort_by)
    for project_url, stats in rows:
        label = _project_label(project_url)
        cells = [Text(label, style="bold")]
        for key, _hdr, _just in _COLUMNS:
            cells.append(_format_cell(stats.get(key, ""), key))
        tbl.add_row(*cells)

    # --- footer ---
    parts: List[str] = []
    if status:
        parts.append(f"[bold]Status[/]: {status}")
    if sleep_reason and sleep_reason.upper() != "NONE":
        parts.append(f"[bold yellow]Sleep[/]: {sleep_reason}")
    if grc_price is not None:
        parts.append(f"[bold]GRC[/]: ${grc_price:.5f}")
    if avg_mag_per_hr is not None:
        parts.append(f"[bold]avg mag/hr[/]: {avg_mag_per_hr:.4f}")
    if hours_user_vs_dev is not None:
        u, d = hours_user_vs_dev
        parts.append(f"[bold]hrs you/dev[/]: {u:.1f}/{d:.1f}")
    if v2_top3:
        top = ", ".join(
            f"[green]{_project_label(u)}[/] [dim]({w:.0f})[/]" for u, w in v2_top3[:3]
        )
        parts.append(f"[bold]v2 top-3[/]: {top}")
    parts.append("[dim]Ctrl-C to exit[/]")
    footer = Text.from_markup("  ·  ".join(parts))

    layout = Group(tbl, Panel(footer, padding=(0, 1), border_style="dim"))
    live.update(Align.left(layout))
