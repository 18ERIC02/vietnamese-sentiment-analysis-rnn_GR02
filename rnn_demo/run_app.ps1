$ErrorActionPreference = 'Stop'

$demoDir = $PSScriptRoot
$pythonPath = Join-Path $demoDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Không tìm thấy Python của môi trường demo: $pythonPath"
}

Push-Location -LiteralPath $demoDir
try {
    & $pythonPath -m streamlit run app.py @args
}
finally {
    Pop-Location
}
