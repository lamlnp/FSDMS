param(
    [ValidateSet('local', 'remote')]
    [string]$Profile = 'local',
    [string]$Host = '0.0.0.0',
    [int]$Port = 8000,
    [bool]$Reload = $true
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$envFile = if ($Profile -eq 'remote') { '.env.remote' } else { '.env' }
$envPath = Join-Path $scriptDir $envFile

if (-not (Test-Path $envPath)) {
    throw "Missing $envFile. Create it from .env.example or .env.remote.example before starting FaceService."
}

$env:FACE_SERVICE_ENV_FILE = $envFile

$python = Join-Path $scriptDir 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}

$uvicornArgs = @('-m', 'uvicorn', 'app.main:app', '--host', $Host, '--port', "$Port")
if ($Reload) {
    $uvicornArgs += '--reload'
}

Write-Host "Starting FaceService with profile '$Profile' using $envFile"
& $python @uvicornArgs
exit $LASTEXITCODE
