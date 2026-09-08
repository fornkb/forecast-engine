"""Terminal output helpers shared by every CLI command."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from time import perf_counter

from rich.console import Console

# Windows terminals default to cp1252, which cannot encode the box-drawing and
# arrow characters Rich and Polars emit.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()


@contextmanager
def log_step(message: str):
    """Print a step banner and its wall-clock duration."""
    console.print(f"[bold cyan]->[/bold cyan] {message}")
    start = perf_counter()
    yield
    console.print(f"   [dim]done in {perf_counter() - start:.2f}s[/dim]")
