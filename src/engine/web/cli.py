"""`engine web` (and `python web.py`): start the local web interface."""

from __future__ import annotations

import threading
import webbrowser
from typing import Annotated

import typer
from rich.console import Console

console = Console()
LOCAL = {"127.0.0.1", "localhost", "::1"}


def serve(
    host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True, preview: bool = False
) -> None:
    try:
        import uvicorn

        from engine.web.app import create_app
    except ImportError as exc:
        console.print(
            f"[red]The web app needs extra packages ({exc.name}).[/red] "
            'Install them with: pip install -e ".[web]"'
        )
        raise SystemExit(1) from exc
    if host not in LOCAL:
        console.print(
            "[yellow]Warning:[/yellow] the web app has no login. Anyone who can reach "
            f"{host}:{port} can run paid AI reviews and edit profiles."
        )
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    if preview:
        console.print(
            "[yellow]Preview mode:[/yellow] sample answers instead of the AI, on a copy of "
            "your data. No API key needed; nothing real is changed."
        )
    app = create_app(preview=preview)
    console.print(f"Forecast reasoning at [bold]{url}[/bold]  (Ctrl+C or close window to stop)")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


def register_web_command(app: typer.Typer) -> None:
    @app.command("web")
    def web(
        host: Annotated[str, typer.Option("--host", help="Address to listen on.")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
        open_browser: Annotated[
            bool, typer.Option("--open/--no-open", help="Open the browser when ready.")
        ] = True,
        preview: Annotated[
            bool,
            typer.Option(
                "--preview", help="Sample answers instead of the AI, on a copy of the data."
            ),
        ] = False,
    ) -> None:
        """Open the reasoning layer in a browser: reviews, findings, profile, AI cost."""
        serve(host, port, open_browser, preview)
