$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PSScriptRoot 'organization_manifest.json'
$plan = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$workspacePath = [System.IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$workspacePrefix = $workspacePath + '\'

function Assert-WorkspacePath([string]$path) {
    $absolute = [System.IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside workspace: $absolute"
    }
    return $absolute
}

# Preflight all paths before the first directory rename.
foreach ($item in $plan.moves) {
    $sourcePath = Assert-WorkspacePath (Join-Path $workspacePath $item.old)
    $targetPath = Assert-WorkspacePath (Join-Path $workspacePath $item.new)
    $source = Get-Item -Force -LiteralPath $sourcePath
    if (-not $source.PSIsContainer -or ($source.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected an original physical directory: $sourcePath"
    }
    if (Test-Path -LiteralPath $targetPath) { throw "Target exists: $targetPath" }
    $parentPath = (Resolve-Path -LiteralPath (Split-Path -Parent $targetPath)).ProviderPath
    Assert-WorkspacePath $parentPath | Out-Null
}

foreach ($item in $plan.moves) {
    $sourcePath = Assert-WorkspacePath (Join-Path $workspacePath $item.old)
    $targetPath = Assert-WorkspacePath (Join-Path $workspacePath $item.new)
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
    try {
        New-Item -ItemType Junction -Path $sourcePath -Target $targetPath | Out-Null
    } catch {
        # Restore the original pathname if compatibility-link creation fails.
        Move-Item -LiteralPath $targetPath -Destination $sourcePath
        throw
    }
    & attrib.exe +H /L $sourcePath
    if ($LASTEXITCODE -ne 0) { throw "Cannot hide compatibility junction: $sourcePath" }
    if ((Get-Item -Force -LiteralPath $targetPath).Attributes -band [IO.FileAttributes]::Hidden) {
        throw "Physical target unexpectedly hidden: $targetPath"
    }
    Add-Content -LiteralPath (Join-Path $PSScriptRoot 'completed_moves.log') -Value $item.old
}

# Remove only the task's temporary probe: junction itself, one file, then empty directories.
$probePath = Assert-WorkspacePath (Join-Path $workspacePath 'organization_junction_probe')
$probeAlias = Assert-WorkspacePath (Join-Path $probePath 'alias')
if ((Get-Item -Force -LiteralPath $probeAlias).LinkType -ne 'Junction') { throw 'Unexpected probe alias' }
[System.IO.Directory]::Delete($probeAlias)
Remove-Item -LiteralPath (Join-Path $probePath 'target/probe.txt')
Remove-Item -LiteralPath (Join-Path $probePath 'target')
Remove-Item -LiteralPath $probePath
Write-Output "Organized $($plan.moves.Count) folders; legacy paths preserved."
