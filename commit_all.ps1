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
    @{Path="README.md"; Msg="Remove README.md"},
    @{Path="augmentations.py"; Msg="Remove augmentations.py"},
    @{Path="confusion_matrices_tiles.csv"; Msg="Remove confusion_matrices_tiles.csv"},
    @{Path="confusion_matrix.py"; Msg="Remove confusion_matrix.py"},
    @{Path="confusion_matrix_aggregate.png"; Msg="Remove confusion_matrix_aggregate.png"},
    @{Path="confusion_matrix_tile_errors.png"; Msg="Remove confusion_matrix_tile_errors.png"},
    @{Path="confusion_matrix_tiles_grid.png"; Msg="Remove confusion_matrix_tiles_grid.png"},
    @{Path="convert_sen12.py"; Msg="Remove convert_sen12.py"},
    @{Path="custom_vocabulary.py"; Msg="Remove custom_vocabulary.py"},
    @{Path="dataset.py"; Msg="Remove dataset.py"},
    @{Path="emergency_bulletin.txt"; Msg="Remove emergency_bulletin.txt"},
    @{Path="final_results.csv"; Msg="Remove final_results.csv"},
    @{Path="generate_prompt.py"; Msg="Remove generate_prompt.py"},
    @{Path="generate_reports.py"; Msg="Remove generate_reports.py"},
    @{Path="geo_lookup.py"; Msg="Remove geo_lookup.py"},
    @{Path="landslide_best_threshold.prev.json"; Msg="Remove landslide_best_threshold.prev.json"},
    @{Path="local_highways.csv"; Msg="Remove local_highways.csv"},
    @{Path="model.py"; Msg="Remove model.py"},
    @{Path="per_image_predictions.csv"; Msg="Remove per_image_predictions.csv"},
    @{Path="per_image_val_results/per_image_val_results.csv"; Msg="Remove per_image_val_results data"},
    @{Path="predict.py"; Msg="Remove predict.py"},
    @{Path="probe_weights.py"; Msg="Remove probe_weights.py"},
    @{Path="spatial_math.py"; Msg="Remove spatial_math.py"},
    @{Path="test_gpu.py"; Msg="Remove test_gpu.py"},
    @{Path="vocabulary.json"; Msg="Remove vocabulary.json"},
    @{Path="train.py"; Msg="Update train.py"},
    @{Path="_migrate.py"; Msg="Add _migrate.py"},
    @{Path="_setup_dirs.py"; Msg="Add _setup_dirs.py"},
    @{Path="data/"; Msg="Add data directory"},
    @{Path="docs/"; Msg="Add docs directory"},
    @{Path="models/"; Msg="Add models directory"},
    @{Path="notebooks/"; Msg="Add notebooks directory"},
    @{Path="results/"; Msg="Add results directory"},
    @{Path="src/"; Msg="Add src directory"},
    @{Path="commit_all.ps1"; Msg="Update commit_all.ps1"}
)

for ($i = 0; $i -lt $items.Count; $i++) {
    Write-Host "=== Commit $($i+1)/$($items.Count): $($items[$i].Path) ==="
    Commit-Item -Path $items[$i].Path -Message $items[$i].Msg
}

Write-Host ""
Write-Host "All $($items.Count) commits completed successfully (no push)."
