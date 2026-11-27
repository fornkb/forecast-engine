"""Open the forecast reasoning web app.

    python web.py              real mode (live AI calls need an API key)
    python web.py --preview    no key needed: sample answers on a copy of the data

Double-clicking this file also works: it restarts itself with the project's
.venv Python, which has the packages installed. Same as `engine web`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        return subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.path.insert(0, str(ROOT / "src"))
    from engine.web.cli import serve

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
    serve(port=port, open_browser="--no-open" not in args, preview="--preview" in args)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 0
    except Exception as exc:  # keep a double-clicked window open long enough to read why
        print(f"\nCould not start the web app: {exc}")
        input("Press Enter to close.")
        code = 1
    sys.exit(code)
