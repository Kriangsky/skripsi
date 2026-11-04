param([switch]$CheckOnly)
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
$expected = @($plan.files.old) + @($plan.retained_root_files) + @($plan.experiment_aliases.old) + @($plan.sections) + @('train','test','.venv-bisindo','__pycache__')
foreach ($entry in Get-ChildItem -LiteralPath $runtimePath -Force) {
    if ($entry.Name -notin $expected) { throw "New runtime item needs manual preservation: $($entry.Name)" }
}
foreach ($item in $plan.files) {
    $target = Inside $item.new
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing section file: $target" }
    if (Test-Path -LiteralPath (Inside $item.old)) {
        if ($item.old -notin @('README.md','.gitignore')) { throw "Unexpected root file: $($item.old)" }
    }
}
foreach ($item in $plan.data) {
    $alias = Get-Item -Force -LiteralPath (Inside $item.old)
    if ($alias.LinkType -ne 'Junction' -or [string]$alias.Target -ne (Inside $item.new)) { throw 'Unexpected dataset alias' }
    $physical = Get-Item -Force -LiteralPath (Inside $item.new)
    if ($physical.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Dataset target is not physical' }
    $mirror = Inside ($plan.runtime + '/' + $item.old)
    $mirrorEntry = Get-Item -Force -LiteralPath $mirror
    if ($mirrorEntry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected data mirror link' }
    $actualFiles = @(Get-ChildItem -Force -File -Recurse -LiteralPath $mirror)
    if ($actualFiles.Count -ne $item.files.Count) { throw 'Data mirror contains new/missing files' }
    foreach ($file in $item.files) {
        $path = Inside ($plan.runtime + '/' + $item.old + '/' + $file.relative)
        $entry = Get-Item -Force -LiteralPath $path
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected mirror file symlink' }
    }
}
foreach ($name in @($plan.experiment_aliases.old) + @($plan.sections) + @('.venv-bisindo')) {
    $entry = Get-Item -Force -LiteralPath (Join-Path $runtimePath $name)
    if ($entry.LinkType -ne 'Junction') { throw "Unexpected runtime directory: $name" }
}
$cacheAlias = Get-Item -Force -LiteralPath (Inside '__pycache__')
if ($cacheAlias.LinkType -ne 'Junction') { throw 'Unexpected cache path' }
foreach ($name in @('README.md','.gitignore')) {
    $backup = Join-Path $PSScriptRoot ('after_rollback_' + $name.TrimStart('.'))
    if (Test-Path -LiteralPath $backup) { throw 'Rollback backup exists; no overwrite' }
}
if ($CheckOnly) { Write-Output 'PASS: rollback preflight; no files moved/deleted.'; return }

foreach ($name in @('README.md','.gitignore')) {
    Move-Item -LiteralPath (Inside $name) -Destination (Join-Path $PSScriptRoot ('after_rollback_' + $name.TrimStart('.')))
}
foreach ($name in @($plan.experiment_aliases.old) + @($plan.sections) + @('.venv-bisindo')) {
    # Nonrecursive deletion removes the junction itself, never its physical target.
    [IO.Directory]::Delete((Join-Path $runtimePath $name))
}
foreach ($item in $plan.files) {
    [IO.File]::Delete((Join-Path $runtimePath $item.old))
    Move-Item -LiteralPath (Inside $item.new) -Destination (Inside $item.old)
}
foreach ($name in $plan.retained_root_files) { [IO.File]::Delete((Join-Path $runtimePath $name)) }
foreach ($item in $plan.data) {
    foreach ($file in $item.files) { [IO.File]::Delete((Inside ($plan.runtime + '/' + $item.old + '/' + $file.relative))) }
    foreach ($directory in ($item.dirs | Sort-Object { $_.Length } -Descending)) {
        [IO.Directory]::Delete((Inside ($plan.runtime + '/' + $item.old + '/' + $directory)))
    }
    [IO.Directory]::Delete((Inside $item.old))
    Move-Item -LiteralPath (Inside $item.new) -Destination (Inside $item.old)
}
[IO.Directory]::Delete((Inside '__pycache__'))
Move-Item -LiteralPath (Inside ($plan.runtime + '/__pycache__')) -Destination (Inside '__pycache__')
[IO.Directory]::Delete($runtimePath)
(Get-Item -Force -LiteralPath (Inside '.venv-bisindo')).Attributes = [IO.FileAttributes]$plan.environment_attributes
foreach ($section in @('01_hasil','02_exp','03_insight_exp')) {
    $readme = Join-Path (Inside $section) 'README.md'
    $text = Get-Content -LiteralPath $readme -Raw
    foreach ($item in $plan.files) {
        $text = $text.Replace(('](../' + $item.new + ')'), ('](../' + $item.old + ')'))
    }
    Set-Content -LiteralPath $readme -Value $text -NoNewline
}
Write-Output 'Original loose-file/data layout restored. New section documents and backed-up navigation remain.'
