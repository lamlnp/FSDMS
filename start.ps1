param(
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8000,
    [bool]$Reload = $true
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$envPath = Join-Path $scriptDir '.env'

if (-not (Test-Path $envPath)) {
    throw "Missing .env. Create it from .env.example before starting FaceService."
}

$python = Join-Path $scriptDir 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}

$uvicornArgs = @('-m', 'uvicorn', 'app.main:app', '--host', $BindHost, '--port', "$Port")
if ($Reload) {
    $uvicornArgs += '--reload'
}

Write-Host "Starting FaceService using .env"
& $python @uvicornArgs
exit $LASTEXITCODE
