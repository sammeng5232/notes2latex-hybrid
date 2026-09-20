"""Entry point for the frozen exe (and `python -m n2lh.launcher`).

Dispatch:
- `notes2latex-hybrid.exe serve ...`    -> CLI server (console behavior)
- `notes2latex-hybrid.exe convert ...`  -> CLI conversion
- `notes2latex-hybrid.exe`              -> desktop GUI window
"""

from __future__ import annotations

import sys

USAGE = """notes2latex-hybrid

  notes2latex-hybrid.exe                 open the desktop GUI
  notes2latex-hybrid.exe serve           run the web server (console)
  notes2latex-hybrid.exe convert FILES   convert files without the GUI
                                         (see: convert --help)

Environment:
  N2LH_PORT           force GUI/server port
  N2LH_GUI_NO_WINDOW  start server without opening a window (automation)
"""


def main() -> int:
    argv = list(sys.argv[1:])
    if argv and argv[0] in ("serve", "convert"):
        from n2lh.cli import main as cli_main
        return cli_main(argv)
    if argv and argv[0] in ("-h", "--help", "help", "/?"):
        sys.stdout.write(USAGE)
        return 0
    from n2lh.gui import run
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
