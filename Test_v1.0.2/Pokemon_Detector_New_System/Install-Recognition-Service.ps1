$ErrorActionPreference = 'Stop'
$Project = 'C:\PokemonDetector'
$TargetDir = Join-Path $Project 'vivaldi_pokemon_extension'
$Source = Join-Path $PSScriptRoot 'service\recognition_service.py'
$Target = Join-Path $TargetDir 'recognition_service.py'
if (-not (Test-Path -LiteralPath $Source)) { throw "Packaged service file not found: $Source" }
if (-not (Test-Path -LiteralPath $TargetDir)) { New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null }
if (Test-Path -LiteralPath $Target) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $backup = "$Target.backup_$stamp"
    Copy-Item -LiteralPath $Target -Destination $backup
    Write-Host "Backed up current service to: $backup" -ForegroundColor Yellow
}
Copy-Item -LiteralPath $Source -Destination $Target -Force
Write-Host "Installed new recognition service: $Target" -ForegroundColor Green
Write-Host 'Your feature databases and source images were not changed.' -ForegroundColor Cyan
Write-Host 'Next, start the service with C:\PokemonDetector\vivaldi_pokemon_extension\Start-PokemonDetectorService.ps1' -ForegroundColor Cyan
