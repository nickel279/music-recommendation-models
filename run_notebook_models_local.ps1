$ErrorActionPreference = "Continue"
$env:USE_NOTEBOOK_MODEL_BRIDGE = "1"
$env:PORT = "5001"
$env:MODEL_DATA_DIR = "model-data-clean"
$env:NOTEBOOK_MODEL_DATA_DIR = "model-data-clean"
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath "C:\Users\nadez\anaconda3\envs\surprise_env\python.exe") {
  & "C:\Users\nadez\anaconda3\envs\surprise_env\python.exe" "app.py"
} else {
  & python "app.py"
}
