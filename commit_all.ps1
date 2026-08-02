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
    @{Path=".gitignore"; Msg="Update .gitignore"},
    @{Path="emergency_bulletin.txt"; Msg="Update emergency_bulletin.txt"},
    @{Path="eval_final.py"; Msg="Update eval_final.py"},
    @{Path="final_results.csv"; Msg="Update final_results.csv"},
    @{Path="generate_prompt.py"; Msg="Update generate_prompt.py"},
    @{Path="generate_reports.py"; Msg="Update generate_reports.py"},
    @{Path="landslide_best_threshold.prev.json"; Msg="Update landslide_best_threshold.prev.json"},
    @{Path="local_highways.csv"; Msg="Remove local_highways.csv"},
    @{Path="predict.py"; Msg="Update predict.py"},
    @{Path="spatial_math.py"; Msg="Update spatial_math.py"},
    @{Path="train.py"; Msg="Update train.py"},
    @{Path="app.py"; Msg="Add app.py"},
    @{Path="commit_all.ps1"; Msg="Add commit_all.ps1"},
    @{Path="custom_vocabulary.py"; Msg="Add custom_vocabulary.py"},
    @{Path="per_image_predictions.csv"; Msg="Add per_image_predictions.csv"},
    @{Path="per_image_val_results/"; Msg="Add per_image_val_results data"},
    @{Path="probe_weights.py"; Msg="Add probe_weights.py"},
    @{Path="vocabulary.json"; Msg="Add vocabulary.json"}
)

for ($i = 0; $i -lt $items.Count; $i++) {
    Write-Host "=== Commit $($i+1)/$($items.Count): $($items[$i].Path) ==="
    Commit-Item -Path $items[$i].Path -Message $items[$i].Msg
}

Write-Host ""
Write-Host "All $($items.Count) commits completed successfully (no push)."
