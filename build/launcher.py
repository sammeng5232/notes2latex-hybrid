"""Thin entry for PyInstaller: delegates to n2lh.launcher."""

from n2lh.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
