param(
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8000,
    [bool]$Reload = $true,
    [switch]$NoNgrok,
    [string]$NgrokPolicyFile = 'ngrok.jwt-validation.yml'
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

$exitCode = 0
$ngrokProcess = $null

try {
    if (-not $NoNgrok) {
        $ngrokCommand = Get-Command ngrok -ErrorAction Stop

        if ([System.IO.Path]::IsPathRooted($NgrokPolicyFile)) {
            $ngrokPolicyPath = $NgrokPolicyFile
        }
        else {
            $ngrokPolicyPath = Join-Path $scriptDir $NgrokPolicyFile
        }

        if (-not (Test-Path $ngrokPolicyPath)) {
            throw "Missing ngrok policy file at '$ngrokPolicyPath'."
        }

        Write-Host "Starting ngrok tunnel with $([System.IO.Path]::GetFileName($ngrokPolicyPath))"
        $ngrokArgs = @('http', "$Port", '--traffic-policy-file', $ngrokPolicyPath)
        $ngrokProcess = Start-Process -FilePath $ngrokCommand.Path -ArgumentList $ngrokArgs -WorkingDirectory $scriptDir -NoNewWindow -PassThru
        Start-Sleep -Seconds 1

        if ($ngrokProcess.HasExited) {
            throw "ngrok exited immediately while starting the tunnel."
        }
    }

    Write-Host "Starting FaceService using .env"
    & $python @uvicornArgs
    $exitCode = $LASTEXITCODE
}
finally {
    if ($ngrokProcess -and -not $ngrokProcess.HasExited) {
        Stop-Process -Id $ngrokProcess.Id -Force
    }
}

exit $exitCode
