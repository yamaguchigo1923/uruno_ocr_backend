# PowerShell helper: create a temporary venv, install requirements and write requirements-locked.txt
param(
    [string]$BackendDir = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\..",
    [string]$VenvName = ".ci_venv"
)

Set-Location -Path $BackendDir
if (Test-Path $VenvName) {
    Write-Host "Removing existing venv $VenvName"
    Remove-Item -Recurse -Force $VenvName
}
python -m venv $VenvName
$py = Join-Path $VenvName "Scripts\python.exe"
& $py -m pip install -U pip
& $py -m pip install -r requirements.txt
& $py -m pip freeze | Out-File -Encoding UTF8 requirements-locked.txt
Write-Host "Wrote requirements-locked.txt"
