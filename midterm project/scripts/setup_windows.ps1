$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } }
py -3.11 -m venv .venv
Check-Exit
$python = Join-Path $PWD '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
Check-Exit
& $python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
Check-Exit
& $python -m pip install -r requirements.txt
Check-Exit
& $python -m pip check
Check-Exit
npm --prefix frontend ci
Check-Exit
npm --prefix frontend run build
Check-Exit
Write-Host 'Setup complete. Start with: .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log'
