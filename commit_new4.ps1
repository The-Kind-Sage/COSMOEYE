$ErrorActionPreference = 'Stop'
$delay = 3
Set-Location $PSScriptRoot

function Commit-Item {
    param([string]$Path, [string]$Message)
    git add -A -- $Path
    if ($LASTEXITCODE -ne 0) { throw "git add failed for $Path" }
    git commit -q -m $Message
    if ($LASTEXITCODE -ne 0) { throw "git commit failed for $Path" }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] Committed: $Message"
    Start-Sleep -Seconds $delay
}

$items = @(
    @{Path="app.py"; Msg="Update app.py"},
    @{Path="generate_prompt.py"; Msg="Update generate_prompt.py"},
    @{Path="generate_reports.py"; Msg="Update generate_reports.py"},
    @{Path="spatial_math.py"; Msg="Update spatial_math.py"}
)

for ($i = 0; $i -lt $items.Count; $i++) {
    Write-Host "=== Commit $($i+1)/$($items.Count): $($items[$i].Path) ==="
    Commit-Item -Path $items[$i].Path -Message $items[$i].Msg
}

Write-Host ""
Write-Host "All $($items.Count) commits completed successfully (no push)."
