$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
$projectPath = [IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$projectPrefix = $projectPath + '\'
function Inside([string]$relative) {
    $path = [IO.Path]::GetFullPath((Join-Path $projectPath $relative))
    if (-not $path.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $path" }
    return $path
}
$runtimePath = Inside $plan.runtime
if (Test-Path -LiteralPath $runtimePath) { throw 'Runtime already exists' }
foreach ($item in $plan.files) {
    $source = Get-Item -Force -LiteralPath (Inside $item.old)
    if ($source.PSIsContainer -or ($source.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Unexpected source: $source" }
    $target = Inside $item.new
    if (Test-Path -LiteralPath $target) { throw "Target exists: $target" }
    $parent = (Resolve-Path -LiteralPath (Split-Path -Parent $target)).ProviderPath
    if (-not $parent.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe target parent' }
}
foreach ($item in $plan.data) {
    $source = Get-Item -Force -LiteralPath (Inside $item.old)
    if (-not $source.PSIsContainer -or ($source.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected dataset source' }
    if (Test-Path -LiteralPath (Inside $item.new)) { throw 'Dataset target exists' }
    Inside $item.new | Out-Null
}
New-Item -ItemType Directory -Path $runtimePath | Out-Null
foreach ($item in $plan.files) {
    $source = Inside $item.old
    $target = Inside $item.new
    Move-Item -LiteralPath $source -Destination $target
    New-Item -ItemType HardLink -Path (Join-Path $runtimePath $item.old) -Target $target | Out-Null
}
foreach ($name in $plan.retained_root_files) {
    New-Item -ItemType HardLink -Path (Join-Path $runtimePath $name) -Target (Inside $name) | Out-Null
}
foreach ($item in $plan.data) {
    $source = Inside $item.old
    $target = Inside $item.new
    Move-Item -LiteralPath $source -Destination $target
    New-Item -ItemType Junction -Path $source -Target $target | Out-Null
    & attrib.exe +H /L $source
    $mirror = Inside ($plan.runtime + '/' + $item.old)
    foreach ($directory in $item.dirs) {
        $path = Inside ($plan.runtime + '/' + $item.old + '/' + $directory)
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    foreach ($file in $item.files) {
        $destination = Inside ($plan.runtime + '/' + $item.old + '/' + $file.relative)
        $original = Inside ($item.new + '/' + $file.relative)
        New-Item -ItemType HardLink -Path $destination -Target $original | Out-Null
    }
}
foreach ($item in $plan.experiment_aliases) {
    New-Item -ItemType Junction -Path (Join-Path $runtimePath $item.old) -Target (Inside $item.new) | Out-Null
}
foreach ($section in $plan.sections) {
    New-Item -ItemType Junction -Path (Join-Path $runtimePath $section) -Target (Inside $section) | Out-Null
}
New-Item -ItemType Junction -Path (Join-Path $runtimePath '.venv-bisindo') -Target (Inside '.venv-bisindo') | Out-Null
$cacheSource = Inside '__pycache__'
$cacheTarget = Inside ($plan.runtime + '/__pycache__')
Move-Item -LiteralPath $cacheSource -Destination $cacheTarget
New-Item -ItemType Junction -Path $cacheSource -Target $cacheTarget | Out-Null
& attrib.exe +H /L $cacheSource
& attrib.exe +H $runtimePath
& attrib.exe +H (Inside '.venv-bisindo')

# Remove the single temporary file-link probe, without recursion.
$probePath = Inside '02_exp/organisasi_20261104/file_link_probe'
Remove-Item -LiteralPath (Join-Path $probePath 'target.txt')
Remove-Item -LiteralPath $probePath
Write-Output "Segmented $($plan.files.Count) files and both datasets; frozen runtime uses hard links, not copied video data."
