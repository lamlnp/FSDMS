param(
    [ValidateSet('auto', 'gpu', 'cpu')]
    [string]$Mode = 'auto',
    [switch]$SkipDeps,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Info($message) {
    Write-Host "    $message"
}

function Write-Warn($message) {
    Write-Host "    $message" -ForegroundColor Yellow
}

# 1. Verify Python >= 3.11
Write-Step "Checking Python"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python not found on PATH. Install Python 3.11+ from https://www.python.org/downloads/windows/ and re-run."
}

$versionOutput = & python --version 2>&1
if ($versionOutput -notmatch 'Python\s+(\d+)\.(\d+)\.(\d+)') {
    throw "Could not parse Python version from '$versionOutput'."
}
$major = [int]$Matches[1]
$minor = [int]$Matches[2]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
    throw "Python $major.$minor detected. FSDMS requires Python 3.11+. Install a newer version and re-run."
}
Write-Info "Found $versionOutput"

# 2. GPU probe + mode selection
Write-Step "Probing for NVIDIA GPU"

$gpuFound = $false
$gpuName = $null
$driverVersion = $null

try {
    $smiOutput = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $smiOutput) {
        $firstLine = ($smiOutput | Select-Object -First 1).Trim()
        $parts = $firstLine -split ',\s*'
        if ($parts.Length -ge 2) {
            $gpuName = $parts[0].Trim()
            $driverVersion = $parts[1].Trim()
            $gpuFound = $true
        }
    }
}
catch {
    # nvidia-smi not present — that's expected on CPU-only machines.
}

if ($gpuFound) {
    Write-Info "Detected GPU: $gpuName (driver $driverVersion) -- GPU mode is supported."
}
else {
    Write-Warn "No NVIDIA GPU detected (nvidia-smi missing or failed). CPU mode recommended."
}

switch ($Mode) {
    'gpu' {
        if (-not $gpuFound) {
            Write-Warn "WARNING: -Mode gpu was requested but nvidia-smi did not report a GPU. Continuing with GPU deps anyway."
        }
        $useGpu = $true
    }
    'cpu' {
        $useGpu = $false
    }
    default {
        $defaultAnswer = if ($gpuFound) { 'Y' } else { 'N' }
        $promptDefault = if ($gpuFound) { '[Y/n]' } else { '[y/N]' }
        $answer = Read-Host "    Use GPU mode? $promptDefault"
        if ([string]::IsNullOrWhiteSpace($answer)) { $answer = $defaultAnswer }
        $useGpu = $answer -match '^[Yy]'
    }
}

$selectedMode = if ($useGpu) { 'GPU' } else { 'CPU' }
Write-Info "Selected mode: $selectedMode"

# 3. Create / reuse venv
Write-Step "Preparing virtual environment"

$venvPath = Join-Path $scriptDir 'venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if ($Force -and (Test-Path $venvPath)) {
    Write-Info "Removing existing venv (-Force)..."
    Remove-Item $venvPath -Recurse -Force
}

if (Test-Path $venvPython) {
    Write-Info "Reusing existing venv at $venvPath"
}
else {
    Write-Info "Creating venv at $venvPath"
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed." }
}

if (-not (Test-Path $venvPython)) {
    throw "Expected venv Python at '$venvPython' but it does not exist."
}

# 4. Install dependencies
if ($SkipDeps) {
    Write-Step "Skipping dependency install (-SkipDeps)"
}
else {
    Write-Step "Installing dependencies"

    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed." }

    if (-not $useGpu) {
        Write-Step "Swapping onnxruntime-gpu -> onnxruntime (CPU mode)"
        & $venvPython -m pip uninstall -y onnxruntime-gpu
        # Don't fail if it wasn't installed in the first place.
        & $venvPython -m pip install onnxruntime
        if ($LASTEXITCODE -ne 0) { throw "pip install onnxruntime failed." }
    }
}

# 5. Create / patch .env
Write-Step "Configuring .env"

$envPath = Join-Path $scriptDir '.env'
$envExamplePath = Join-Path $scriptDir '.env.example'

$envCreated = $false
if (-not (Test-Path $envPath)) {
    if (-not (Test-Path $envExamplePath)) {
        throw "Missing .env.example -- cannot bootstrap .env."
    }
    Copy-Item $envExamplePath $envPath
    Write-Info "Created .env from template."
    $envCreated = $true
}
else {
    Write-Info ".env exists -- leaving values intact (only GPU_DEVICE_ID will be updated)."
}

$gpuValue = if ($useGpu) { '0' } else { '-1' }
$lines = [System.IO.File]::ReadAllLines($envPath)
$found = $false
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match '^\s*GPU_DEVICE_ID\s*=') {
        $lines[$i] = "GPU_DEVICE_ID=$gpuValue"
        $found = $true
        break
    }
}
if (-not $found) {
    $lines += "GPU_DEVICE_ID=$gpuValue"
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($envPath, $lines, $utf8NoBom)
Write-Info "Set GPU_DEVICE_ID=$gpuValue"

# 6. Final summary
Write-Step "Setup complete"
Write-Host ""
Write-Host "  Mode:    $selectedMode" -ForegroundColor Green
Write-Host "  venv:    $venvPath" -ForegroundColor Green
Write-Host "  .env:    $envPath $(if ($envCreated) { '(new)' } else { '(updated)' })" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env to set FACE_SERVICE_API_KEY (must match BEDMS)."
Write-Host "  2. Run:  .\start.ps1 -NoNgrok"
Write-Host ""
