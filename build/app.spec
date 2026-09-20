# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for notes2latex-hybrid (onefile, windowed)."""

import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

hiddenimports = [
    # uvicorn loads protocol/loop implementations dynamically
    *collect_submodules("uvicorn"),
    # our own package (entry is a thin launcher)
    "n2lh", "n2lh.api", "n2lh.cli", "n2lh.config", "n2lh.export", "n2lh.gui",
    "n2lh.jobs", "n2lh.launcher", "n2lh.main", "n2lh.store",
    "n2lh.compiler.latex",
    "n2lh.pipeline.assembler", "n2lh.pipeline.autofix", "n2lh.pipeline.context",
    "n2lh.pipeline.figures", "n2lh.pipeline.graph", "n2lh.pipeline.ingest",
    "n2lh.pipeline.layout", "n2lh.pipeline.sanitize", "n2lh.pipeline.segment",
    "n2lh.pipeline.style", "n2lh.pipeline.tables",
    "n2lh.recognition.base", "n2lh.recognition.heuristic",
    "n2lh.recognition.prompts", "n2lh.recognition.trocr", "n2lh.recognition.vlm",
    # desktop window / tray fallbacks
    "webview", "webview.platforms.edgechromium", "webview.platforms.winforms",
    "pystray", "pystray._win32",
]

datas = [
    ("../n2lh/web/static", "n2lh/web/static"),
]

binaries = []
try:  # pdfium native library (pdf support)
    binaries += collect_dynamic_libs("pypdfium2")
except Exception:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath("..")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="notes2latex-hybrid",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # windowed GUI exe (CLI passthrough still works)
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon="icon.ico",
)
