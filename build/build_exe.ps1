# Build notes2latex-hybrid.exe (onefile, windowed) with PyInstaller.
# Usage:  powershell -File build\build_exe.ps1
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $root
try {
    # A running frozen app locks dist\notes2latex-hybrid.exe and breaks the build.
    Get-Process notes2latex-hybrid -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    Write-Host "[build] installing build deps into .venv (non-editable: PyInstaller cannot trace editable installs)"
    uv pip install --python .venv\Scripts\python.exe --reinstall-package notes2latex-hybrid ".[build,pdf,dev]" --quiet
    if (-not $SkipTests) {
        Write-Host "[build] running test suite first"
        & .venv\Scripts\python.exe -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "tests failed; aborting build" }
    }

    Write-Host "[build] generating icon + favicon"
    & .venv\Scripts\python.exe build\make_icon.py
    if ($LASTEXITCODE -ne 0) { throw "icon generation failed" }
    Copy-Item build\icon.ico n2lh\web\static\favicon.ico -Force

    Write-Host "[build] running PyInstaller"
    & .venv\Scripts\pyinstaller.exe --noconfirm --clean --workpath build\work --distpath dist build\app.spec
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

    # Fail fast on missing modules we actually need.
    $warnFile = "build\work\app\warn-app.txt"
    if (Test-Path $warnFile) {
        $critical = Select-String -Path $warnFile -Pattern "missing module named '(n2lh|webview|pystray|pystray\._win32|pypdfium2|uvicorn)[.\w']*"
        if ($critical) {
            $critical | ForEach-Object { Write-Warning $_.Line }
            throw "critical modules missing from the frozen app (see warnings above)"
        }
    }

    # Strict-cleanup hygiene: remove PyInstaller intermediates, keep dist/ + spec + launcher + icon.
    if (Test-Path build\work) { Remove-Item build\work -Recurse -Force }

    $exe = Join-Path $root "dist\notes2latex-hybrid.exe"
    $size = "{0:N1}" -f ((Get-Item $exe).Length / 1MB)
    Write-Host "[build] done: $exe ($size MB)"
} finally {
    Pop-Location
}
