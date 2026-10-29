$ErrorActionPreference = 'Stop'
$Project = 'C:\PokemonDetector'
$Service = Join-Path $Project 'vivaldi_pokemon_extension copy\recognition_service.py'
if (-not (Test-Path -LiteralPath $Service)) {
    throw "Service file not found: $Service. Copy the package files first."
}
Set-Location $Project
Write-Host 'Starting local Pokémon recognition service on 127.0.0.1:8765' -ForegroundColor Cyan
Write-Host 'Keep this PowerShell window open while using the extension.' -ForegroundColor Yellow
python $Service
