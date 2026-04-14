Set-Location -Path "$PSScriptRoot"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-Not (Test-Path $python)) {
    Write-Error ".venv\Scripts\python.exe not found. Activate your virtual environment or install dependencies first."
    exit 1
}

& $python -m streamlit run ckd_dashboard.py --server.port 8501 --server.headless true


