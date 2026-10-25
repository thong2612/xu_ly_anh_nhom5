param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

& $Python -m src.dataset.cli pipeline
exit $LASTEXITCODE
