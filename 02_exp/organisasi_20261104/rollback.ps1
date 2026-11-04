$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'organization_manifest.json') -Raw | ConvertFrom-Json
$workspacePath = [System.IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$workspacePrefix = $workspacePath + '\'

function Assert-WorkspacePath([string]$path) {
    $absolute = [System.IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside workspace: $absolute"
    }
    return $absolute
}

foreach ($item in $plan.moves) {
    $legacyPath = Assert-WorkspacePath (Join-Path $workspacePath $item.old)
    $physicalPath = Assert-WorkspacePath (Join-Path $workspacePath $item.new)
    $alias = Get-Item -Force -LiteralPath $legacyPath
    if ($alias.LinkType -ne 'Junction' -or [string]$alias.Target -ne $physicalPath) {
        throw "Unexpected compatibility link: $legacyPath"
    }
    if (-not (Test-Path -LiteralPath $physicalPath -PathType Container)) {
        throw "Physical directory missing: $physicalPath"
    }
    Assert-WorkspacePath (Resolve-Path -LiteralPath $physicalPath).ProviderPath | Out-Null
}

foreach ($item in $plan.moves) {
    $legacyPath = Assert-WorkspacePath (Join-Path $workspacePath $item.old)
    $physicalPath = Assert-WorkspacePath (Join-Path $workspacePath $item.new)
    # Directory.Delete without recursion removes only the junction, never its target.
    [System.IO.Directory]::Delete($legacyPath)
    Move-Item -LiteralPath $physicalPath -Destination $legacyPath
}
Write-Output 'Original folder paths restored. Section indexes remain as historical navigation.'
