param(
    [switch]$Tests,
    [switch]$Notebook,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectPath '.compat_runtime'
$pythonPath = Join-Path $projectPath '.venv-bisindo/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtimePath -PathType Container)) { throw 'Compatibility runtime missing' }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Project Python environment missing' }
& $pythonPath (Join-Path $PSScriptRoot 'check_runtime.py') $projectPath
if ($LASTEXITCODE -ne 0) { throw 'Compatibility check failed; no experiment was started.' }
Push-Location -LiteralPath $runtimePath
try {
    if ($Tests) { & $pythonPath -m unittest discover -v }
    elseif ($Notebook) { & $pythonPath -m jupyterlab --notebook-dir=$runtimePath }
    elseif ($PythonArgs.Count -gt 0) { & $pythonPath @PythonArgs }
    else { throw 'Pass Python arguments, -Tests, or -Notebook. Nothing was executed.' }
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
