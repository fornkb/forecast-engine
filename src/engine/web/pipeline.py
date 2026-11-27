"""Run the base engine's forecast and backtest from the web app.

Calls the very same `engine forecast` / `engine backtest` command functions in
this process (no duplicated logic), captures their terminal output for the page,
and runs one at a time: both are CPU-heavy and write the client's store.
"""

from __future__ import annotations

import threading

import click
import typer

from engine.reason.schemas import ReasoningError

_LOCK = threading.Lock()


def run_engine_command(command: str, client_id: str, **options) -> str:
    from engine import cli  # imported lazily: it loads the forecasting libraries
    from engine.utils.console import console

    function = {"forecast": cli.forecast, "backtest": cli.backtest}[command]
    defaults = {
        "forecast": {"horizon": 3, "model": None, "export": None},
        "backtest": {"horizon": None, "origins": None, "max_history": None},
    }[command]
    failed = 0
    with _LOCK:
        with console.capture() as capture:
            try:
                function(client=client_id, **{**defaults, **options})
            except (click.exceptions.Exit, typer.Exit) as exc:
                failed = getattr(exc, "exit_code", 1)
            except SystemExit as exc:
                failed = exc.code or 0
        output = capture.get()
    if failed:
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise ReasoningError(f"engine {command} stopped:\n{tail}")
    return output
