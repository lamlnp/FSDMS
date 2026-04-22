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

        Write-Host "Starting ngrok tunnel with $([System.IO.Path]::GetFileName($ngrokPolicyPath)) in a new window..."
        $ngrokArgs = @('http', "$Port", '--traffic-policy-file', $ngrokPolicyPath)
        
        # Removed -NoNewWindow so ngrok spawns in its own terminal
        $ngrokProcess = Start-Process -FilePath $ngrokCommand.Path -ArgumentList $ngrokArgs -WorkingDirectory $scriptDir -PassThru
        Start-Sleep -Seconds 1

        if ($ngrokProcess.HasExited) {
            throw "ngrok exited immediately while starting the tunnel."
        }
    }

    Write-Host "Starting FaceService using .env in the main window..."
    & $python @uvicornArgs
    $exitCode = $LASTEXITCODE
}
finally {
    # This will still properly clean up the second window when you close the main one
    if ($ngrokProcess -and -not $ngrokProcess.HasExited) {
        Write-Host "Shutting down ngrok tunnel..."
        Stop-Process -Id $ngrokProcess.Id -Force
    }
}

exit $exitCode