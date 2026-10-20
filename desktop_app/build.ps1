# PowerShell build script for packaging HCMUTE RNN Sentiment Desktop Application into Windows EXE
param (
    [switch]$Console = $false, # Pass -Console to keep debug console window visible
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   HCMUTE RNN SENTIMENT ANALYSIS - DESKTOP BUILD SCRIPT   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Determine Python and PyInstaller executable
$VENV_PYTHON = Join-Path $SCRIPT_DIR "..\rnn_demo\.venv\Scripts\python.exe"
if (Test-Path $VENV_PYTHON) {
    $PYTHON = (Resolve-Path $VENV_PYTHON).Path
} else {
    $PYTHON = (Get-Command python.exe).Source
}

Write-Host "Using Python: $PYTHON" -ForegroundColor Green

# 2. Clean previous build artifacts
if ($Clean) {
    Write-Host "Cleaning previous build and dist directories..." -ForegroundColor Yellow
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
}

# 3. Assemble PyInstaller arguments
$APP_NAME = "RNN_Sentiment_HCMUTE"
$MAIN_SCRIPT = "main.py"
$ICON_PATH = "assets\hcmute_logo.png"

$PYINSTALLER_ARGS = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--name", $APP_NAME,
    "--paths", ".",
    "--add-data", "assets;assets",
    "--add-data", "models;models",
    "--hidden-import", "ui",
    "--hidden-import", "ui.main_window",
    "--hidden-import", "ui.styles",
    "--hidden-import", "ui.widgets",
    "--hidden-import", "inference",
    "--hidden-import", "inference.predictor",
    "--hidden-import", "PySide6",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--hidden-import", "tensorflow",
    "--hidden-import", "keras",
    "--collect-data", "keras",
    "--copy-metadata", "keras",
    "--copy-metadata", "tensorflow"
)

if (Test-Path $ICON_PATH) {
    $PYINSTALLER_ARGS += @("--icon", $ICON_PATH)
}

if (-not $Console) {
    $PYINSTALLER_ARGS += @("--windowed")
}

$PYINSTALLER_ARGS += @($MAIN_SCRIPT)

Write-Host "Executing PyInstaller..." -ForegroundColor Green
& $PYTHON $PYINSTALLER_ARGS

# 4. Copy assets, models, and module mirrors directly into dist directory to guarantee accessibility
$DIST_APP_DIR = Join-Path $SCRIPT_DIR "dist\$APP_NAME"
if (Test-Path $DIST_APP_DIR) {
    Write-Host "Ensuring assets and models are in distribution root..." -ForegroundColor Green
    Copy-Item -Path "assets" -Destination (Join-Path $DIST_APP_DIR "assets") -Recurse -Force
    Copy-Item -Path "models" -Destination (Join-Path $DIST_APP_DIR "models") -Recurse -Force

    # Mirror desktop_app package inside _internal so both 'desktop_app.ui' and 'ui' imports resolve from any CWD
    $INTERNAL_DIR = Join-Path $DIST_APP_DIR "_internal"
    if (Test-Path $INTERNAL_DIR) {
        $INTERNAL_PKG = Join-Path $INTERNAL_DIR "desktop_app"
        New-Item -ItemType Directory -Force -Path $INTERNAL_PKG | Out-Null
        New-Item -ItemType File -Force -Path (Join-Path $INTERNAL_PKG "__init__.py") | Out-Null
        Copy-Item -Path "ui" -Destination (Join-Path $INTERNAL_PKG "ui") -Recurse -Force
        Copy-Item -Path "inference" -Destination (Join-Path $INTERNAL_PKG "inference") -Recurse -Force
    }

    $EXE_PATH = Join-Path $DIST_APP_DIR "$APP_NAME.exe"
    if (Test-Path $EXE_PATH) {
        Write-Host "`nBUILD SUCCESSFUL!" -ForegroundColor Green
        Write-Host "Executable location:" -ForegroundColor Cyan
        Write-Host "  $EXE_PATH" -ForegroundColor White
        Write-Host "`nYou can run the application directly by double-clicking the EXE file." -ForegroundColor Yellow
    } else {
        Write-Error "Build finished but executable was not found at: $EXE_PATH"
    }
} else {
    Write-Error "PyInstaller build directory not created at: $DIST_APP_DIR"
}
